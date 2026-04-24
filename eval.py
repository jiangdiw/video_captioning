import argparse
import json
import os

import torch
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from models.captioning_model import CaptioningModel, PrecomputedCaptioningModel, SequencePrecomputedCaptioningModel
from models.multimodal_encoder import Approach1Encoder, Approach2Encoder
from data.dataset import (
    MSRVTTDataset,
    PrecomputedDataset,
    SequencePrecomputedDataset,
    collate_fn,
    collate_fn_precomputed,
    collate_fn_sequence_precomputed,
)
from data.vocabulary import Vocabulary
from data.msrvtt import normalize_dataset_mode
from torch.utils.data import DataLoader
from tqdm import tqdm


DEFAULT_EXTERNAL_MSRVTT_ROOT = "/Volumes/Seagate Portable Drive/west_point_backup_v2/AY25-1/MA498/TASCC_MSRVTT"
DEFAULT_EXTERNAL_MSRVTT_ROOT_V3 = "/Volumes/Seagate Portable Drive/west_point_backup_v3/AY25-1/MA498/TASCC_MSRVTT"


def choose_default_path(*candidates):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(candidate)
        if parent and os.path.exists(parent) and os.access(parent, os.W_OK):
            return candidate
    return candidates[-1]


def ensure_exists(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(opt):
    import misc.utils as utils  # noqa: F401
    from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel

    if opt["model"] == "S2VTModel":
        return S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            dim_vid=opt["dim_vid"],
            rnn_cell=opt["rnn_type"],
            n_layers=opt["num_layers"],
            rnn_dropout_p=opt["rnn_dropout_p"],
        )
    if opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            batch_size=opt["batch_size"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt["rnn_type"],
            rnn_dropout_p=opt["rnn_dropout_p"],
            n_layers=opt["num_layers"],
        )
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            n_layers=opt["num_layers"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt["rnn_type"],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"],
        )
        return S2VTAttModel(encoder, decoder)
    raise ValueError(f"Unsupported model: {opt['model']}")


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append({"image_id": row[1], "cap_id": len(gts[row[1]]), "caption": row[0]})
        else:
            gts[row[1]] = [{"image_id": row[1], "cap_id": 0, "caption": row[0]}]
    return gts


def convert_caption_dict_to_coco_scorer_format(captions):
    gts = {}
    for video_id, caps in captions.items():
        gts[video_id] = [
            {"image_id": video_id, "cap_id": idx, "caption": caption}
            for idx, caption in enumerate(caps)
        ]
    return gts


def load_ground_truth_captions(path):
    with open(path) as f:
        payload = json.load(f)

    # Processed caption format: {video_id: [caption1, caption2, ...]}
    if isinstance(payload, dict) and "sentences" not in payload:
        return convert_caption_dict_to_coco_scorer_format(payload)

    # Raw MSR-VTT metadata format with a top-level "sentences" list.
    if isinstance(payload, dict) and "sentences" in payload:
        grouped = {}
        for sentence in payload["sentences"]:
            video_id = sentence["video_id"]
            grouped.setdefault(video_id, []).append(
                {
                    "image_id": video_id,
                    "cap_id": len(grouped.get(video_id, [])),
                    "caption": sentence["caption"],
                }
            )
        return grouped

    raise ValueError(f"Unsupported caption ground-truth format in {path}")


def test(model, dataset, vocab, opt, device):
    import misc.utils as utils
    from pandas import json_normalize

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=opt["batch_size"],
        shuffle=False,
        num_workers=opt.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )
    scorer = COCOScorer()
    gt_dataframe = json_normalize(json.load(open(opt["eval_json"]))["sentences"])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)

    results = []
    samples = {}

    for data in tqdm(loader):
        fc_feats = data["fc_feats"].to(device)
        video_ids = data["video_ids"]
        with torch.no_grad():
            _, seq_preds = model(fc_feats, mode="inference", opt=opt)

        sents = utils.decode_sequence(vocab, seq_preds)
        for index, sent in enumerate(sents):
            video_id = video_ids[index]
            samples[video_id] = [{"image_id": video_id, "caption": sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print("VALID SCORE:", valid_score)
    os.makedirs(opt["results_path"], exist_ok=True)

    with open(os.path.join(opt["results_path"], "scores.txt"), "a") as scores_table:
        scores_table.write(json.dumps(results[0]) + "\n")
    with open(
        os.path.join(opt["results_path"], opt["model"].split("/")[-1].split(".")[0] + ".json"),
        "w",
    ) as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score}, prediction_results)


def build_end_to_end_model(opt):
    feature_mode = opt["feature_mode"]
    variant = opt["variant"]
    if feature_mode == "precomputed":
        return PrecomputedCaptioningModel(
            input_dim=opt["input_dim"],
            vocab_size=opt["vocab_size"],
            embed_dim=opt["embed_dim"],
            hidden_dim=opt["hidden_dim"],
            dropout=opt["dropout"],
        )
    if feature_mode == "precomputed_sequence":
        return SequencePrecomputedCaptioningModel(
            seq_input_dim=opt["input_dim"],
            pooled_input_dim=opt["pooled_input_dim"],
            vocab_size=opt["vocab_size"],
            embed_dim=opt["embed_dim"],
            hidden_dim=opt["hidden_dim"],
            dropout=opt["dropout"],
        )
    if feature_mode == "raw":
        encoder = Approach1Encoder() if variant == "approach1" else Approach2Encoder()
        return CaptioningModel(
            encoder=encoder,
            vocab_size=opt["vocab_size"],
            embed_dim=opt["embed_dim"],
            hidden_dim=opt["hidden_dim"],
            dropout=opt["dropout"],
        )
    raise ValueError(f"Unsupported end-to-end feature_mode: {feature_mode}")


def test_end_to_end(model, dataset, vocab, opt, device):
    if opt["feature_mode"] == "precomputed":
        loader = DataLoader(
            dataset,
            batch_size=opt.get("batch_size", 64),
            shuffle=False,
            num_workers=opt.get("num_workers", 0),
            collate_fn=collate_fn_precomputed,
            pin_memory=device.type == "cuda",
        )
    elif opt["feature_mode"] == "precomputed_sequence":
        loader = DataLoader(
            dataset,
            batch_size=opt.get("batch_size", 64),
            shuffle=False,
            num_workers=opt.get("num_workers", 0),
            collate_fn=collate_fn_sequence_precomputed,
            pin_memory=device.type == "cuda",
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=opt.get("batch_size", 64),
            shuffle=False,
            num_workers=opt.get("num_workers", 0),
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )

    gts = load_ground_truth_captions(opt["test_caption_json"])
    print(
        f"Loaded ground truth from {opt['test_caption_json']} "
        f"with {len(gts)} video ids"
    )

    scorer = COCOScorer()
    model.eval()
    samples = {}

    for batch in tqdm(loader):
        if opt["feature_mode"] == "precomputed":
            embeddings, _, video_ids = batch
            embeddings = embeddings.to(device)
            with torch.no_grad():
                seq_preds = model.generate(
                    embeddings,
                    max_len=opt.get("generation_max_len", 20),
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                )
        elif opt["feature_mode"] == "precomputed_sequence":
            seq_embeddings, pooled_embeddings, _, video_ids = batch
            seq_embeddings = seq_embeddings.to(device)
            pooled_embeddings = pooled_embeddings.to(device)
            with torch.no_grad():
                seq_preds = model.generate(
                    seq_embeddings,
                    pooled_embeddings,
                    max_len=opt.get("generation_max_len", 20),
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                )
        else:
            clips, dinos, audios, _, _, video_ids = batch
            clips = clips.to(device)
            dinos = dinos.to(device)
            audios = audios.to(device)
            with torch.no_grad():
                seq_preds = model.generate(
                    clips,
                    dinos,
                    audios,
                    max_len=opt.get("generation_max_len", 20),
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                )

        sents = [vocab.decode(seq.tolist()) for seq in seq_preds]
        for index, sent in enumerate(sents):
            video_id = video_ids[index]
            samples[video_id] = [{"image_id": video_id, "caption": sent}]

    sample_ids = set(samples.keys())
    gt_ids = set(gts.keys())
    missing_gt = sorted(sample_ids - gt_ids)
    if missing_gt:
        preview = ", ".join(missing_gt[:10])
        raise KeyError(
            f"Predictions include {len(missing_gt)} video_ids not present in ground truth "
            f"(gt_path={opt['test_caption_json']}, gt_count={len(gt_ids)}, pred_count={len(sample_ids)}): {preview}"
        )

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    print("VALID SCORE:", valid_score)
    os.makedirs(opt["results_path"], exist_ok=True)

    with open(os.path.join(opt["results_path"], "scores.txt"), "a") as scores_table:
        scores_table.write(json.dumps(valid_score) + "\n")
    with open(os.path.join(opt["results_path"], f"{opt['model_name']}.json"), "w") as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score}, prediction_results)


def main(opt):
    if opt.get("framework") == "end_to_end":
        ensure_exists(opt["vocab_path"], "vocab_path")
        ensure_exists(opt["test_caption_json"], "test_caption_json")
        if not opt.get("saved_model"):
            opt["saved_model"] = opt.get("default_saved_model")
        ensure_exists(opt["saved_model"], "saved_model checkpoint")
        ensure_exists(opt["recover_opt"], "recover_opt json")
        os.makedirs(opt["results_path"], exist_ok=True)

        device = resolve_device(opt.get("device", "auto"))
        vocab = Vocabulary.load(opt["vocab_path"])
        opt["vocab_size"] = len(vocab)
        dataset_mode = normalize_dataset_mode(opt.get("dataset_mode", "subset"))
        processed_root = opt.get("processed_root")
        print(
            f"Evaluating end-to-end model_name={opt.get('model_name')} "
            f"dataset_mode={dataset_mode} processed_root={processed_root} "
            f"feature_mode={opt.get('feature_mode')}"
        )

        if opt["feature_mode"] == "precomputed":
            dataset = PrecomputedDataset(
                "test",
                vocab,
                variant=opt["variant"],
                dataset_mode=dataset_mode,
                processed_root=processed_root,
            )
        elif opt["feature_mode"] == "precomputed_sequence":
            dataset = SequencePrecomputedDataset(
                "test",
                vocab,
                variant=opt["variant"],
                dataset_mode=dataset_mode,
                processed_root=processed_root,
            )
        else:
            dataset = MSRVTTDataset("test", vocab, dataset_mode=dataset_mode, processed_root=processed_root)

        model = build_end_to_end_model(opt).to(device)
        state = torch.load(opt["saved_model"], map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        print(f"Evaluating end-to-end model on device: {device}")
        test_end_to_end(model, dataset, vocab, opt, device)
        return

    ensure_exists(opt["info_json"], "info_json")
    ensure_exists(opt["caption_json"], "caption_json")
    ensure_exists(opt["saved_model"], "saved_model checkpoint")
    ensure_exists(opt["recover_opt"], "recover_opt json")
    ensure_exists(opt["eval_json"], "evaluation ground-truth json")
    for feats_dir in opt["feats_dir"]:
        ensure_exists(feats_dir, "feature directory")

    from dataloader_v2 import VideoDataset as LegacyVideoDataset

    device = resolve_device(opt["device"])
    dataset = LegacyVideoDataset(opt, "test")
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
    model = build_model(opt).to(device)
    state = torch.load(opt["saved_model"], map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    print(f"Evaluating on device: {device}")
    test(model, dataset, dataset.get_vocab(), opt, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recover_opt",
        type=str,
        default=choose_default_path(
            os.path.join(DEFAULT_EXTERNAL_MSRVTT_ROOT, "save_tascc", "opt_info.json"),
            os.path.join(DEFAULT_EXTERNAL_MSRVTT_ROOT_V3, "save_tascc", "opt_info.json"),
            "save_tascc/opt_info.json",
        ),
        help="recover train opts from saved opt_json",
    )
    parser.add_argument(
        "--saved_model",
        type=str,
        default=None,
        help="path to saved model to evaluate",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=None, help="minibatch size")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--eval_json",
        type=str,
        default=None,
        help="ground-truth json used for scoring the test split",
    )
    parser.add_argument("--sample_max", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--dataset-mode", type=str, default=None, choices=["subset", "full"])

    args = vars(parser.parse_args())
    ensure_exists(args["recover_opt"], "recover_opt json")
    opt = json.load(open(args["recover_opt"]))
    opt["recover_opt"] = args["recover_opt"]

    for key, value in args.items():
        if key == "recover_opt":
            continue
        if value is not None:
            opt[key] = value

    if opt.get("framework") == "end_to_end":
        checkpoint_dir = opt.get("checkpoint_dir")
        if not opt.get("saved_model"):
            if opt.get("default_saved_model"):
                opt["saved_model"] = opt["default_saved_model"]
            elif checkpoint_dir:
                opt["saved_model"] = os.path.join(checkpoint_dir, "best.pth")
        if not opt.get("results_path"):
            if checkpoint_dir:
                opt["results_path"] = os.path.join(
                    os.path.dirname(checkpoint_dir), "..", "results", opt["model_name"]
                )
                opt["results_path"] = os.path.normpath(opt["results_path"])
            else:
                opt["results_path"] = os.path.join("outputs", "results", opt["model_name"])
        opt.setdefault("batch_size", 64)
        opt.setdefault("num_workers", 0)
        opt.setdefault("device", "auto")
        opt.setdefault("generation_max_len", 20)
    else:
        if not opt.get("saved_model"):
            opt["saved_model"] = choose_default_path(
                os.path.join(DEFAULT_EXTERNAL_MSRVTT_ROOT, "save_tascc", "model_200.pth"),
                os.path.join(DEFAULT_EXTERNAL_MSRVTT_ROOT_V3, "save_tascc", "model_200.pth"),
                "save_tascc/model_200.pth",
            )
        if not opt.get("results_path"):
            opt["results_path"] = choose_default_path(
                os.path.join(DEFAULT_EXTERNAL_MSRVTT_ROOT, "results_tascc"),
                os.path.join(DEFAULT_EXTERNAL_MSRVTT_ROOT_V3, "results_tascc"),
                "results_tascc/",
            )
        opt.setdefault("batch_size", 64)
        opt.setdefault("num_workers", 0)
        opt.setdefault("device", "auto")
        opt.setdefault("eval_json", "dataset/MSR-VTT/test_videodatainfo.json")
        opt.setdefault("sample_max", 1)
        opt.setdefault("temperature", 1.0)
        opt.setdefault("beam_size", 1)

    main(opt)
