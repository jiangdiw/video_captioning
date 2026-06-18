import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(scriptDir, "..");
const outDir = path.join(projectDir, "submission_artifacts", "reproducibility_package");
const cleanRunRoot = "<DATTALION_WORKSPACE>/runs_clean/transfer_arch_search_20260610";
const cleanLvecrDir = path.join(cleanRunRoot, "long_video_evidence_reranker_adaptive_visualsupport_nomargin_v2");
const cleanDirectDir = "<DATTALION_WORKSPACE>/runs_clean/size_74_freeze_adaptive120_v1/finecap20_nr2";
const sourceZeroShotDir = "<DATTALION_WORKSPACE>/runs/size_0";

const manifest = [];
const missing = [];

const denyFileNames = new Set([
  ".DS_Store",
  "Thumbs.db",
]);

const denySuffixes = [
  ".aux",
  ".bbl",
  ".blg",
  ".fdb_latexmk",
  ".fls",
  ".log",
  ".out",
];

const rawMediaSuffixes = [
  ".avi",
  ".m4v",
  ".mkv",
  ".mov",
  ".mp4",
  ".mpeg",
  ".mpg",
  ".webm",
];

const textLikeSuffixes = new Set([
  ".bib",
  ".bst",
  ".cls",
  ".csv",
  ".json",
  ".md",
  ".mjs",
  ".py",
  ".tex",
  ".txt",
]);

const pathSanitizers = [
  ["external_dattalion_videos", "external_dattalion_videos"],
  [
    "<PROJECT_ROOT>",
    "<PROJECT_ROOT>",
  ],
  ["<DATTALION_WORKSPACE>", "<DATTALION_WORKSPACE>"],
  ["<DATTALION_ASSETS>", "<DATTALION_ASSETS>"],
  ["<VIDEO_CAPTIONING_ROOT>", "<VIDEO_CAPTIONING_ROOT>"],
  ["<PYTHON_BIN>", "<PYTHON_BIN>"],
  ["<MSRVTT_ROOT>", "<MSRVTT_ROOT>"],
  ["<VIDEO_SUMMARIZATION_ROOT>", "<VIDEO_SUMMARIZATION_ROOT>"],
];

const scriptFiles = [
  "tools/build_reproducibility_package.mjs",
];

const coreResultDirs = [];

const splitMetadataFiles = [
  "submission_artifacts/splits/audit.json",
  "submission_artifacts/splits/splits.json",
];

const manuscriptFiles = [
  "sn-article-resubmission-draft.tex",
  "sn-article-resubmission-draft.pdf",
  "sn-bibliography.bib",
  "sn-jnl.cls",
  "sn-mathphys-num.bst",
];

const manuscriptFigureFiles = [
  "resubmission_figures/source_target_duration_shift.png",
  "resubmission_figures/network_architecture.pdf",
  "resubmission_figures/ablation_summary.png",
  "resubmission_figures/dattalion_transfer_curve_clean_lvecr.png",
  "resubmission_figures/repeated_sampling_cider.png",
  "resubmission_figures/bootstrap_clean_lvecr.png",
  "resubmission_figures/clean_baseline_stress_lvecr.png",
  "resubmission_figures/dattalion_transfer_ablation_cider.png",
  "resubmission_figures/domain_keyword_coverage_clean_lvecr.png",
  "resubmission_figures/reference_agreement.png",
];

const normalizedReferenceFiles = [
  "normalized_references_v2/STYLE_GUIDE.md",
  "normalized_references_v2/summary.json",
];

function relToProject(absPath) {
  return path.relative(projectDir, absPath).split(path.sep).join("/");
}

function shouldSkip(filePath) {
  const base = path.basename(filePath);
  if (denyFileNames.has(base)) return true;
  if (base.startsWith("~$")) return true;
  if (filePath.split(path.sep).includes("__pycache__")) return true;
  const lower = base.toLowerCase();
  if (denySuffixes.some((suffix) => lower.endsWith(suffix))) return true;
  if (rawMediaSuffixes.some((suffix) => lower.endsWith(suffix))) return true;
  return false;
}

function isTextLike(filePath) {
  return textLikeSuffixes.has(path.extname(filePath).toLowerCase());
}

function sanitizeLocalPaths(text) {
  let sanitized = text;
  for (const [needle, replacement] of pathSanitizers) {
    sanitized = sanitized.replaceAll(needle, replacement);
  }
  return sanitized;
}

async function sha256(absPath) {
  const data = await fs.readFile(absPath);
  return crypto.createHash("sha256").update(data).digest("hex");
}

async function ensureCleanOutDir() {
  const expectedParent = path.join(projectDir, "submission_artifacts");
  const resolvedOut = path.resolve(outDir);
  if (!resolvedOut.startsWith(`${path.resolve(expectedParent)}${path.sep}`)) {
    throw new Error(`Refusing to clean unexpected output directory: ${resolvedOut}`);
  }
  await fs.rm(outDir, { recursive: true, force: true });
  await fs.mkdir(outDir, { recursive: true });
}

async function copyFile(srcRel, destRel, category, note = "") {
  const src = path.join(projectDir, srcRel);
  const dest = path.join(outDir, destRel);
  console.error(`copy ${srcRel} -> ${destRel}`);
  try {
    await fs.stat(src);
  } catch {
    missing.push(srcRel);
    return;
  }
  if (shouldSkip(src)) return;
  await fs.mkdir(path.dirname(dest), { recursive: true });
  if (isTextLike(src)) {
    const text = await fs.readFile(src, "utf8");
    await fs.writeFile(dest, sanitizeLocalPaths(text));
  } else {
    await fs.copyFile(src, dest);
  }
  const stat = await fs.stat(dest);
  console.error(`hash ${destRel}`);
  manifest.push({
    path: destRel.split(path.sep).join("/"),
    category,
    bytes: stat.size,
    sha256: await sha256(dest),
    source: sanitizeLocalPaths(srcRel),
    note,
  });
}

async function copyExternalFile(srcAbs, destRel, category, note = "") {
  const src = path.resolve(srcAbs);
  const dest = path.join(outDir, destRel);
  console.error(`copy ${sanitizeLocalPaths(src)} -> ${destRel}`);
  try {
    await fs.stat(src);
  } catch {
    missing.push(sanitizeLocalPaths(src));
    return;
  }
  if (shouldSkip(src)) return;
  await fs.mkdir(path.dirname(dest), { recursive: true });
  if (isTextLike(src)) {
    const text = await fs.readFile(src, "utf8");
    await fs.writeFile(dest, sanitizeLocalPaths(text));
  } else {
    await fs.copyFile(src, dest);
  }
  const stat = await fs.stat(dest);
  console.error(`hash ${destRel}`);
  manifest.push({
    path: destRel.split(path.sep).join("/"),
    category,
    bytes: stat.size,
    sha256: await sha256(dest),
    source: sanitizeLocalPaths(src),
    note,
  });
}

async function copyDir(srcRel, destRel, category) {
  const srcDir = path.join(projectDir, srcRel);
  let entries;
  try {
    entries = await fs.readdir(srcDir, { withFileTypes: true });
  } catch {
    missing.push(srcRel);
    return;
  }

  for (const entry of entries) {
    const src = path.join(srcDir, entry.name);
    const nextSrcRel = relToProject(src);
    const nextDestRel = path.join(destRel, entry.name);
    if (shouldSkip(src)) continue;
    if (entry.isDirectory()) {
      await copyDir(nextSrcRel, nextDestRel, category);
    } else if (entry.isFile()) {
      await copyFile(nextSrcRel, nextDestRel, category);
    }
  }
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '"' && inQuotes && next === '"') {
      field += '"';
      i += 1;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      row.push(field);
      field = "";
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(field);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function csvEscape(value) {
  const s = value == null ? "" : String(value);
  return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function toCsv(rows) {
  return `${rows.map((row) => row.map(csvEscape).join(",")).join("\n")}\n`;
}

async function writeFileTracked(destRel, contents, category, source = "generated by tools/build_reproducibility_package.mjs", note = "") {
  const dest = path.join(outDir, destRel);
  await fs.mkdir(path.dirname(dest), { recursive: true });
  await fs.writeFile(dest, contents);
  const stat = await fs.stat(dest);
  manifest.push({
    path: destRel.split(path.sep).join("/"),
    category,
    bytes: stat.size,
    sha256: await sha256(dest),
    source,
    note,
  });
}

async function writeJsonTracked(destRel, value, category, source, note = "") {
  await writeFileTracked(destRel, `${JSON.stringify(value, null, 2)}\n`, category, source, note);
}

async function readJson(absPath) {
  return JSON.parse(await fs.readFile(absPath, "utf8"));
}

function metricValue(metrics, key) {
  return Number(metrics[key] ?? 0);
}

function metricRow(label, metrics, note = "") {
  return {
    method: label,
    Bleu_1: metricValue(metrics, "Bleu_1"),
    Bleu_2: metricValue(metrics, "Bleu_2"),
    Bleu_3: metricValue(metrics, "Bleu_3"),
    Bleu_4: metricValue(metrics, "Bleu_4"),
    METEOR: metricValue(metrics, "METEOR"),
    ROUGE_L: metricValue(metrics, "ROUGE_L"),
    CIDEr: metricValue(metrics, "CIDEr"),
    note,
  };
}

function fixedRow(label, values, note = "") {
  return {
    method: label,
    Bleu_1: values.Bleu_1,
    Bleu_2: values.Bleu_2,
    Bleu_3: values.Bleu_3 ?? null,
    Bleu_4: values.Bleu_4 ?? null,
    METEOR: values.METEOR,
    ROUGE_L: values.ROUGE_L,
    CIDEr: values.CIDEr,
    note,
  };
}

function rowsToCsv(rows, columns) {
  return toCsv([
    columns,
    ...rows.map((row) => columns.map((column) => row[column])),
  ]);
}

function metricMarkdown(rows, columns) {
  const header = `| ${columns.join(" | ")} |`;
  const sep = `| ${columns.map(() => "---").join(" | ")} |`;
  const body = rows.map((row) => `| ${columns.map((column) => row[column] ?? "").join(" | ")} |`);
  return `${[header, sep, ...body].join("\n")}\n`;
}

async function copyCurrentManuscriptFigures() {
  for (const file of manuscriptFigureFiles) {
    await copyFile(file, `manuscript/${file}`, "manuscript figures");
  }
}

async function writeCleanPredictionArtifacts() {
  await copyExternalFile(
    path.join(sourceZeroShotDir, "test_predictions.json"),
    "results/predictions/source_zero_shot_test_predictions.json",
    "predictions",
    "Zero-shot source-model Dattalion predictions."
  );
  await copyExternalFile(
    path.join(cleanDirectDir, "test_predictions.json"),
    "results/predictions/adaptive120_direct_test_predictions.json",
    "predictions",
    "Strongest direct-transfer adaptive120 predictions used for the manuscript table."
  );
  await copyExternalFile(
    path.join(cleanLvecrDir, "test_predictions.json"),
    "results/predictions/lvecr_test_predictions.json",
    "predictions",
    "Clean LV-ECR predictions under model-score candidate ordering."
  );
  await copyExternalFile(
    path.join(cleanLvecrDir, "test_oracle_predictions.json"),
    "results/predictions/lvecr_candidate_pool_oracle_predictions.json",
    "predictions",
    "Diagnostic candidate-pool oracle predictions; not a deployable system."
  );
  await copyFile(
    "baseline_stress_results/blip_predictions.json",
    "results/predictions/blip_frame_caption_test_predictions.json",
    "predictions",
    "Three-frame BLIP control predictions."
  );
  await copyFile(
    "baseline_stress_results/git_predictions.json",
    "results/predictions/git_frame_caption_test_predictions.json",
    "predictions",
    "Three-frame GIT control predictions."
  );

  const lvecrPredictions = await readJson(path.join(cleanLvecrDir, "test_predictions.json"));
  const baselinePredictions = lvecrPredictions.map((row) => ({
    video_id: row.video_id,
    caption: row.baseline_caption,
    sentence_cider: row.baseline_sentence_cider,
    references: row.references,
    note: "Clean adapted model-score baseline derived from the LV-ECR candidate pool before evidence reranking.",
  }));
  await writeJsonTracked(
    "results/predictions/clean_modelscore_baseline_test_predictions.json",
    baselinePredictions,
    "predictions",
    path.join(cleanLvecrDir, "test_predictions.json"),
    "Extracted baseline captions and sentence-CIDEr values from the clean LV-ECR run."
  );
}

async function writeCleanResultSummaries() {
  const lvecrSummary = await readJson(path.join(cleanLvecrDir, "summary.json"));
  const bootstrap = await readJson(path.join(cleanLvecrDir, "bootstrap_sentence_cider.json"));
  const sourceMetrics = await readJson(path.join(sourceZeroShotDir, "test_metrics.json"));
  const directMetrics = await readJson(path.join(cleanDirectDir, "test_metrics.json"));

  await copyExternalFile(
    path.join(sourceZeroShotDir, "test_metrics.json"),
    "results/metrics/source_zero_shot_test_metrics.json",
    "metrics",
    "Zero-shot source-model metrics on Dattalion."
  );
  await copyExternalFile(
    path.join(cleanDirectDir, "test_metrics.json"),
    "results/metrics/adaptive120_direct_test_metrics.json",
    "metrics",
    "Strongest direct-transfer adaptive120 metrics."
  );
  await copyExternalFile(
    path.join(cleanLvecrDir, "summary.json"),
    "results/lvecr/lvecr_summary.json",
    "LV-ECR",
    "Authoritative clean LV-ECR run summary."
  );
  await copyExternalFile(
    path.join(cleanLvecrDir, "bootstrap_sentence_cider.json"),
    "results/lvecr/lvecr_bootstrap_sentence_cider.json",
    "LV-ECR",
    "Paired sentence-CIDEr bootstrap for clean adapted baseline versus LV-ECR."
  );
  await copyExternalFile(
    path.join(cleanLvecrDir, "weight_grid_top50.json"),
    "results/lvecr/lvecr_weight_grid_top50.json",
    "LV-ECR",
    "Top clean LV-ECR development-set weight settings."
  );

  const mainRows = [
    fixedRow("0-shot source only", { Bleu_1: 0.3702, Bleu_2: 0.1472, METEOR: 0.1367, ROUGE_L: 0.2009, CIDEr: 0.0491 }, "Manuscript rounded row; full source metrics are packaged separately."),
    fixedRow("10-shot direct", { Bleu_1: 0.4389, Bleu_2: 0.1857, METEOR: 0.1975, ROUGE_L: 0.2710, CIDEr: 0.0921 }, "Manuscript rounded row."),
    fixedRow("20-shot direct", { Bleu_1: 0.4591, Bleu_2: 0.2027, METEOR: 0.1926, ROUGE_L: 0.2652, CIDEr: 0.1031 }, "Manuscript rounded row."),
    fixedRow("30-shot direct", { Bleu_1: 0.4640, Bleu_2: 0.2141, METEOR: 0.1786, ROUGE_L: 0.2689, CIDEr: 0.0804 }, "Manuscript rounded row."),
    fixedRow("74-shot adaptive120 direct", { Bleu_1: 0.6224, Bleu_2: 0.3624, METEOR: 0.2790, ROUGE_L: 0.3336, CIDEr: 0.1477 }, "Rounded from the packaged adaptive120 direct metrics."),
    fixedRow("Clean adapted model-score baseline", { Bleu_1: 0.5962, Bleu_2: 0.3324, METEOR: 0.2744, ROUGE_L: 0.3232, CIDEr: 0.1374 }, "Rounded from the clean LV-ECR summary baseline metrics."),
    fixedRow("LV-ECR (adaptive visual+support)", { Bleu_1: 0.6009, Bleu_2: 0.3440, METEOR: 0.2885, ROUGE_L: 0.3198, CIDEr: 0.1521 }, "Rounded from the clean LV-ECR summary metrics."),
    fixedRow("LV-ECR candidate-pool oracle", { Bleu_1: 0.7815, Bleu_2: 0.5055, METEOR: 0.3776, ROUGE_L: 0.4204, CIDEr: 0.3205 }, "Rounded from the clean LV-ECR diagnostic oracle metrics."),
  ];
  const mainColumns = ["method", "Bleu_1", "Bleu_2", "METEOR", "ROUGE_L", "CIDEr", "note"];
  await writeJsonTracked("results/metrics/main_transfer_table_clean_lvecr.json", mainRows, "metrics", "sn-article-resubmission-draft.tex");
  await writeFileTracked("results/metrics/main_transfer_table_clean_lvecr.csv", rowsToCsv(mainRows, mainColumns), "metrics", "sn-article-resubmission-draft.tex");
  await writeFileTracked("results/metrics/main_transfer_table_clean_lvecr.md", metricMarkdown(mainRows, mainColumns), "metrics", "sn-article-resubmission-draft.tex");

  const fullPrecisionRows = [
    metricRow("source_zero_shot", sourceMetrics, "Full-precision source metrics from the executable run."),
    metricRow("adaptive120_direct", directMetrics, "Full-precision direct-transfer metrics from finecap20_nr2."),
    metricRow("clean_modelscore_baseline", lvecrSummary.test_baseline_metrics, "Full-precision clean baseline metrics from the LV-ECR summary."),
    metricRow("lvecr", lvecrSummary.test_metrics, "Full-precision clean LV-ECR metrics."),
    metricRow("lvecr_candidate_pool_oracle", lvecrSummary.test_oracle_metrics, "Full-precision diagnostic oracle metrics."),
  ];
  const precisionColumns = ["method", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "note"];
  await writeJsonTracked("results/metrics/full_precision_clean_lvecr_metrics.json", fullPrecisionRows, "metrics", "clean LV-ECR run artifacts");
  await writeFileTracked("results/metrics/full_precision_clean_lvecr_metrics.csv", rowsToCsv(fullPrecisionRows, precisionColumns), "metrics", "clean LV-ECR run artifacts");

  const stressRows = [
    { method: "Caption prior medoid", uses_test_video: false, uses_target_train_captions: true, uses_external_pretraining: false, Bleu_2: 0.1143, METEOR: 0.1741, ROUGE_L: 0.1775, CIDEr: 0.0186, keyword_coverage: "100.0%" },
    { method: "Train-neighbor retrieval", uses_test_video: true, uses_target_train_captions: true, uses_external_pretraining: true, Bleu_2: 0.1267, METEOR: 0.2021, ROUGE_L: 0.2109, CIDEr: 0.0364, keyword_coverage: "90.0%" },
    { method: "BLIP frame-caption baseline", uses_test_video: true, uses_target_train_captions: false, uses_external_pretraining: true, Bleu_2: 0.1729, METEOR: 0.2085, ROUGE_L: 0.2361, CIDEr: 0.0827, keyword_coverage: "53.3%" },
    { method: "GIT frame-caption baseline", uses_test_video: true, uses_target_train_captions: false, uses_external_pretraining: true, Bleu_2: 0.1790, METEOR: 0.1949, ROUGE_L: 0.2378, CIDEr: 0.0711, keyword_coverage: "43.3%" },
    { method: "Source-only transfer", uses_test_video: true, uses_target_train_captions: false, uses_external_pretraining: true, Bleu_2: 0.1472, METEOR: 0.1367, ROUGE_L: 0.2009, CIDEr: 0.0489, keyword_coverage: "16.7%" },
    { method: "Strongest direct transfer", uses_test_video: true, uses_target_train_captions: true, uses_external_pretraining: true, Bleu_2: 0.3624, METEOR: 0.2790, ROUGE_L: 0.3336, CIDEr: 0.1477, keyword_coverage: "83.3%" },
    { method: "Clean adapted model-score baseline", uses_test_video: true, uses_target_train_captions: true, uses_external_pretraining: true, Bleu_2: 0.3324, METEOR: 0.2744, ROUGE_L: 0.3232, CIDEr: 0.1374, keyword_coverage: "90.0%" },
    { method: "LV-ECR", uses_test_video: true, uses_target_train_captions: true, uses_external_pretraining: true, Bleu_2: 0.3440, METEOR: 0.2885, ROUGE_L: 0.3198, CIDEr: 0.1521, keyword_coverage: "93.3%" },
  ];
  const stressColumns = ["method", "uses_test_video", "uses_target_train_captions", "uses_external_pretraining", "Bleu_2", "METEOR", "ROUGE_L", "CIDEr", "keyword_coverage"];
  await writeJsonTracked("results/baseline_stress/clean_baseline_stress_summary.json", stressRows, "baseline stress", "sn-article-resubmission-draft.tex");
  await writeFileTracked("results/baseline_stress/clean_baseline_stress_summary.csv", rowsToCsv(stressRows, stressColumns), "baseline stress", "sn-article-resubmission-draft.tex");
  await writeFileTracked("results/baseline_stress/clean_baseline_stress_summary.md", metricMarkdown(stressRows, stressColumns), "baseline stress", "sn-article-resubmission-draft.tex");

  const variantRows = [
    { variant: "Clean model-score baseline", METEOR: 0.2744, ROUGE_L: 0.3232, CIDEr: 0.1374 },
    { variant: "LV-ECR, unconstrained grid", METEOR: 0.2725, ROUGE_L: 0.3090, CIDEr: 0.1397 },
    { variant: "LV-ECR, visual-positive", METEOR: 0.2741, ROUGE_L: 0.3122, CIDEr: 0.1443 },
    { variant: "LV-ECR, visual+support+margin", METEOR: 0.2622, ROUGE_L: 0.3109, CIDEr: 0.1467 },
    { variant: "LV-ECR, visual+support", METEOR: 0.2885, ROUGE_L: 0.3198, CIDEr: 0.1521 },
  ];
  const variantColumns = ["variant", "METEOR", "ROUGE_L", "CIDEr"];
  await writeJsonTracked("results/transfer_ablation/clean_lvecr_variant_summary.json", variantRows, "transfer ablation", "sn-article-resubmission-draft.tex");
  await writeFileTracked("results/transfer_ablation/clean_lvecr_variant_summary.csv", rowsToCsv(variantRows, variantColumns), "transfer ablation", "sn-article-resubmission-draft.tex");
  await writeFileTracked("results/transfer_ablation/clean_lvecr_variant_summary.md", metricMarkdown(variantRows, variantColumns), "transfer ablation", "sn-article-resubmission-draft.tex");

  const bootSummary = {
    n_bootstrap: bootstrap.n_bootstrap,
    seed: bootstrap.seed,
    corpus_baseline_CIDEr: bootstrap.corpus_baseline_metrics.CIDEr,
    corpus_lvecr_CIDEr: bootstrap.corpus_method_metrics.CIDEr,
    paired_sentence_cider_diff: bootstrap.bootstrap_paired_sentence_cider_diff,
    interpretation: "The paired mean sentence-CIDEr gain is positive, but the 95% interval narrowly crosses zero.",
  };
  await writeJsonTracked("results/robustness/clean_lvecr_bootstrap_summary.json", bootSummary, "robustness", path.join(cleanLvecrDir, "bootstrap_sentence_cider.json"));
  const bootMd = `# Clean LV-ECR Bootstrap Summary

- Baseline corpus CIDEr: ${bootstrap.corpus_baseline_metrics.CIDEr}
- LV-ECR corpus CIDEr: ${bootstrap.corpus_method_metrics.CIDEr}
- Paired bootstrap mean sentence-CIDEr gain: ${bootstrap.bootstrap_paired_sentence_cider_diff.mean}
- 95% interval: [${bootstrap.bootstrap_paired_sentence_cider_diff.p2_5}, ${bootstrap.bootstrap_paired_sentence_cider_diff.p97_5}]
- Interpretation: positive average gain, but the interval narrowly crosses zero.
`;
  await writeFileTracked("results/robustness/clean_lvecr_bootstrap_summary.md", bootMd, "robustness", path.join(cleanLvecrDir, "bootstrap_sentence_cider.json"));
}

async function writeSanitizedExperimentIndex() {
  const packageRows = [
    ["experiment_id", "section", "purpose", "split_description", "train_videos", "val_videos", "test_videos", "seed_or_draws", "selection_protocol", "metric_script", "primary_artifact"],
    ["msrvtt_full_source_bart", "Source-domain model", "Train the full MSR-VTT source captioner", "MSR-VTT full split", "6513", "497", "2990", "1337", "Best CIDEr XE checkpoint on MSR-VTT validation", "external_training_workspace/video_captioning/train_final_bart.py", "not_packaged_source_training_output"],
    ["dattalion_zero_shot", "Transfer baseline", "Evaluate source model zero-shot on Dattalion", "Cleaned Dattalion executable split", "0", "10", "30", "1337", "No target adaptation; source checkpoint frozen", "code/scripts/evaluate_dattalion_checkpoint.py", "results/predictions/source_test_predictions.json"],
    ["dattalion_direct_20", "Direct target adaptation", "Measure low-resource direct transfer with 20 target training videos", "Cleaned Dattalion executable split", "20", "10", "30", "1337", "Best-loss checkpoint reevaluated on held-out test split", "external_training_workspace/video_captioning/train_final_bart.py", "not_packaged_model_training_output"],
    ["dattalion_direct_adaptive120", "Direct target adaptation", "Measure strongest direct transfer with longer-video adaptive coverage", "Cleaned Dattalion executable split", "74", "10", "30", "1337", "Best held-out direct checkpoint from adaptive120 sweep", "external_training_workspace/video_captioning/train_final_bart.py", "results/predictions/direct_test_predictions.json"],
    ["dattalion_lexical_top5", "Target lexical adaptation", "Lightweight target-side lexical adaptation with curated captions", "Cleaned Dattalion executable split", "74", "10", "30", "1337", "Five-epoch partial lexical-only BART tuning using top-5 medoid captions/video", "external_training_workspace/video_captioning/train_final_bart.py", "not_packaged_model_training_config"],
    ["dattalion_clean_modelscore_baseline", "Target lexical adaptation", "Evaluate the clean adapted model-score candidate selector", "Cleaned Dattalion executable split", "74", "10", "30", "1337", "BART beam candidates ordered by generation model score; no stored CIDEr ordering used", "code/scripts/run_long_video_evidence_reranker.py", "results/predictions/clean_modelscore_baseline_test_predictions.json"],
    ["dattalion_lvecr", "Final transfer system", "Apply long-video evidence-calibrated reranking to clean model-score-ordered candidates", "Cleaned Dattalion executable split", "84 usable non-test", "0", "30", "20260610", "Weights selected on train+validation non-test videos with positive visual and support constraints; evaluated once on held-out test", "code/scripts/run_long_video_evidence_reranker.py", "results/lvecr/lvecr_summary.json"],
    ["dattalion_lvecr_oracle", "Diagnostic oracle", "Measure candidate-pool headroom for the clean LV-ECR candidate pool", "Cleaned Dattalion executable split", "0", "0", "30", "20260610", "Per-video highest-CIDEr candidate selected after evaluation; not a deployable system", "code/scripts/run_long_video_evidence_reranker.py", "results/predictions/lvecr_candidate_pool_oracle_predictions.json"],
    ["dattalion_repeated_sampling", "Robustness", "Repeated low-resource subset sampling on direct transfer", "Cleaned Dattalion executable split", "10/20/30 subsets", "10", "30", "5 random draws per size", "Repeated draws from the fixed 74-video train pool with fixed val/test split", "code/scripts/run_dattalion_repeated_sampling.py", "results/robustness/repeated_sampling_low_resource/repeated_sampling_summary.md"],
    ["dattalion_lvecr_bootstrap", "Robustness", "Bootstrap uncertainty for the clean adapted baseline versus LV-ECR", "Cleaned Dattalion executable split", "0", "0", "30", "10000 bootstrap resamples", "Paired bootstrap over held-out test videos using per-video sentence-CIDEr", "code/scripts/run_long_video_evidence_reranker.py", "results/robustness/clean_lvecr_bootstrap_summary.md"],
    ["dattalion_baseline_stress", "Control experiments", "Stress-test retrieval priors, external frame captioners, clean adapted baseline, and LV-ECR", "Cleaned Dattalion executable split", "84 usable non-test where needed", "0", "30", "1337 / 20260610", "Fixed protocol; clean LV-ECR rows use model-score candidate ordering and held-out test evaluation", "code/scripts/run_dattalion_baseline_stress_tests.py; code/scripts/run_long_video_evidence_reranker.py", "results/baseline_stress/clean_baseline_stress_summary.json"],
    ["dattalion_lvecr_variants", "Transfer ablation", "Compare clean LV-ECR feature-constraint variants", "Cleaned Dattalion executable split", "84 usable non-test", "0", "30", "20260610", "Variant weights selected on non-test videos and evaluated on held-out test", "code/scripts/run_long_video_evidence_reranker.py", "results/transfer_ablation/clean_lvecr_variant_summary.json"],
    ["dattalion_reference_normalization", "Evaluation sensitivity", "Measure how lexical normalization changes agreement and system scores", "Cleaned Dattalion executable split", "0", "0", "30", "rule-based normalization v2", "Reference-only normalization applied before rescoring; system ranking checked for invariance", "code/scripts/normalize_dattalion_references_v2.py", "data/normalized_references_v2/summary.md"],
  ];

  await writeFileTracked(
    "docs/experiment_index.sanitized.csv",
    toCsv(packageRows),
    "documentation",
    "experiment_index.csv",
    "Absolute local paths replaced by package-relative paths or external-workspace placeholders."
  );
}

async function writeReferenceNotes() {
  const readme = `# Reproducibility Package

This package contains the derived artifacts for the revised multimedia-journal submission on transfer learning for long-video captioning under Dattalion domain shift.

It intentionally does not include raw Dattalion videos or local training workspaces. The raw videos have separate platform-level access and redistribution conditions. Files that require raw video access use the placeholder directory \`external_dattalion_videos/\`.

## Contents

- \`manuscript/\`: submitted TeX source, bibliography, Springer class/style files, generated PDF, and only the figures referenced by the current manuscript.
- \`data/submission_artifacts/splits/\`: cleaned executable Dattalion split metadata.
- \`data/normalized_references_v2/\`: normalized-reference sensitivity summary and style guide.
- \`results/predictions/\`: held-out Dattalion test predictions for the source, adaptive120 direct-transfer, clean adapted model-score baseline, LV-ECR, LV-ECR candidate-pool oracle, BLIP, and GIT systems.
- \`results/metrics/\`: manuscript-synchronized transfer table plus full-precision clean LV-ECR metric summaries.
- \`results/lvecr/\`: authoritative clean LV-ECR summary, bootstrap file, and development-set weight-grid artifact.
- \`results/robustness/\`: clean LV-ECR paired-bootstrap summary.
- \`results/baseline_stress/\`: retrieval, caption-prior, external frame-captioner, clean adapted baseline, and LV-ECR controls.
- \`results/transfer_ablation/\`: clean LV-ECR variant summary.
- \`code/\`: the package-builder script used to assemble this derived-artifact bundle. Full training/evaluation scripts remain in the local project workspace and are not bundled here because several Drive-backed historical helper files are not reliably readable in the current sync state.
- \`docs/\`: sanitized experiment index and reproducibility notes.

## Main Audit Path

1. Inspect \`docs/experiment_index.sanitized.csv\` for the experiment protocol, split sizes, selection rules, and primary packaged artifacts.
2. Inspect \`results/metrics/main_transfer_table_clean_lvecr.md\` for the manuscript-synchronized transfer table.
3. Inspect \`results/lvecr/lvecr_summary.json\` and \`results/lvecr/lvecr_bootstrap_sentence_cider.json\` for the clean LV-ECR result, oracle headroom, selected weights, and bootstrap interval.
4. Inspect \`results/baseline_stress/\` and \`results/transfer_ablation/\` for the controls and LV-ECR variant summaries used in the revised manuscript.
5. Verify file integrity with \`shasum -a 256 -c CHECKSUMS.sha256\` from this directory.

## Regeneration Notes

- The analysis scripts assume the broader local training environment used for the paper, including the source video-captioning repository, trained checkpoints, Python metric dependencies, and externally obtained Dattalion videos.
- The package is designed so that reviewers can audit the final derived results without raw-video redistribution.
- Rebuilding this package from the project root can be done with \`node tools/build_reproducibility_package.mjs\`.
- Refreshing the upload archive after a rebuild can be done from \`submission_artifacts/\` with \`zip -qr reproducibility_package.zip reproducibility_package\`.

## Integrity Files

- \`MANIFEST.json\`: package-relative path, category, byte count, SHA-256 checksum, and source/provenance note for each packaged file.
- \`CHECKSUMS.sha256\`: standard checksum file for command-line verification.
`;

  await writeFileTracked("README.md", readme, "documentation");

  const notes = `# Reproducibility Notes

## Scope

This is a derived-artifact package, not a full raw-data mirror. It supports review of the reported predictions, clean LV-ECR metrics, robustness tables, normalized-reference sensitivity analysis, stress tests, figures, and manuscript source.

## Raw Media

Raw Dattalion videos are excluded. The human-evaluation sanitized sheet uses \`external_dattalion_videos/<video_id>.mp4\` placeholders so an authorized user can map those rows to their own video copy.

## Private or Internal Files Excluded

Internal reviewer-strategy notes and draft planning files are intentionally excluded. TeX build intermediates, cache folders, temporary spreadsheet lock files, and raw video formats are also excluded.

## Clean LV-ECR Result

The current manuscript does not report the earlier single-rater blinded human-evaluation result. That material is intentionally excluded from this upload package. The final reported system is LV-ECR under clean model-score candidate ordering; the paired bootstrap interval for the gain over the clean adapted baseline is included under \`results/lvecr/\` and summarized under \`results/robustness/\`.
`;

  await writeFileTracked("docs/REPRODUCIBILITY_NOTES.md", notes, "documentation");
}

async function writeManifestFiles() {
  manifest.sort((a, b) => a.path.localeCompare(b.path));
  const manifestPath = path.join(outDir, "MANIFEST.json");
  await fs.writeFile(manifestPath, `${JSON.stringify({ generated_at: new Date().toISOString(), file_count: manifest.length, files: manifest }, null, 2)}\n`);

  const checksumLines = [];
  for (const entry of manifest) {
    checksumLines.push(`${entry.sha256}  ${entry.path}`);
  }
  await fs.writeFile(path.join(outDir, "CHECKSUMS.sha256"), `${checksumLines.join("\n")}\n`);
}

async function main() {
  await ensureCleanOutDir();

  for (const file of manuscriptFiles) {
    await copyFile(file, `manuscript/${path.basename(file)}`, "manuscript");
  }
  await copyCurrentManuscriptFigures();

  for (const [src, dest, category] of coreResultDirs) {
    await copyDir(src, dest, category);
  }
  for (const file of splitMetadataFiles) {
    await copyFile(file, `data/${file}`, "split metadata");
  }
  for (const file of normalizedReferenceFiles) {
    await copyFile(file, `data/${file}`, "normalized references");
  }

  for (const file of scriptFiles) {
    await copyFile(file, `code/${file}`, "code");
  }

  await writeCleanPredictionArtifacts();
  await writeCleanResultSummaries();
  await writeSanitizedExperimentIndex();
  await writeReferenceNotes();
  await writeManifestFiles();

  if (missing.length > 0) {
    console.error("Missing expected files/directories:");
    for (const item of missing) console.error(`- ${item}`);
    process.exitCode = 1;
  } else {
    console.log(`Built ${outDir}`);
    console.log(`Packaged ${manifest.length} files plus MANIFEST.json and CHECKSUMS.sha256`);
  }
}

await main();
