const form = document.querySelector("#generate-form");
const duration = document.querySelector("#duration");
const durationValue = document.querySelector("#duration-value");
const statusPill = document.querySelector("#status-pill");
const generateButton = document.querySelector("#generate-button");
const buttonText = generateButton.querySelector(".button-text");
const kaggleJobBox = document.querySelector("#kaggle-job");
const downloads = document.querySelector("#downloads");
const audioSlot = document.querySelector("#audio-slot");
const warningSlot = document.querySelector("#warning-slot");
const lyricsOutput = document.querySelector("#lyrics-output");
const canvas = document.querySelector("#wave-canvas");
const ctx = canvas.getContext("2d");
let isGenerating = false;
let activePollId = 0;
let currentJob = null;
const DEFAULT_TTS_MODEL = "hynt/F5-TTS-Vietnamese-ViVoice";
const DEFAULT_MMS_TTS_MODEL = "facebook/mms-tts-vie";
const DEFAULT_TTS_VOICE = "f5_vietnamese_vivoice_reference";
const DEFAULT_CUSTOM_MUSIC_MODEL = "genmusic-vn/custom-symbolic-composer";

duration.addEventListener("input", () => {
  durationValue.textContent = `Mục tiêu ${duration.value} giây`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isGenerating) return;

  setGenerating(true, "Đang gửi...");
  statusPill.textContent = "Đang gửi";
  downloads.innerHTML = "";
  renderWarning(null);
  audioSlot.textContent = "Đang chờ kết quả custom composer từ Kaggle...";
  lyricsOutput.textContent = "Đang chuẩn bị lời bài hát và kế hoạch vocal...";
  kaggleJobBox.textContent = "";

  const data = {
    text: document.querySelector("#text").value,
    duration_seconds: Number(duration.value),
    genre: document.querySelector("#genre").value,
    model: DEFAULT_CUSTOM_MUSIC_MODEL,
  };

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const job = await response.json();
    if (!response.ok || job.error) {
      throw new Error(job.error || "Không thể gửi job");
    }
    renderJob(job);
    statusPill.textContent = job.status === "needs_setup" ? "Cần cấu hình Kaggle" : "Kaggle";
    if (!["needs_setup", "failed", "complete"].includes(job.status)) {
      pollKaggle(job.run_id);
    } else {
      setGenerating(false);
    }
  } catch (error) {
    statusPill.textContent = "Lỗi";
    kaggleJobBox.textContent = error.message;
    setGenerating(false);
  }
});

downloads.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-retry-tts]");
  if (!button || isGenerating) return;
  await retryTts(button.getAttribute("data-retry-tts"));
});

function setGenerating(active, label = "Tạo MP3") {
  isGenerating = active;
  generateButton.disabled = active;
  generateButton.classList.toggle("is-loading", active);
  generateButton.setAttribute("aria-busy", active ? "true" : "false");
  buttonText.textContent = label;
}

function renderJob(job) {
  currentJob = job;
  const durationPlan = job.duration_plan || {};
  const targetDuration = job.target_duration_seconds || durationPlan.target_duration_seconds;
  const plannedBacking = job.planned_backing_duration_seconds || durationPlan.planned_backing_duration_seconds;
  const durationCeiling = job.duration_ceiling_seconds || durationPlan.duration_ceiling_seconds;
  const outroTail = job.outro_tail_seconds || durationPlan.outro_tail_seconds;
  const lines = [
    `Trạng thái: ${formatJobStatus(job.status)}`,
    `Model custom: ${job.model || DEFAULT_CUSTOM_MUSIC_MODEL}`,
    `TTS: ${job.tts_model || DEFAULT_TTS_MODEL}`,
    `TTS dự phòng: ${job.mms_tts_model || DEFAULT_MMS_TTS_MODEL}`,
    `Giọng TTS: ${job.tts_voice_actual || DEFAULT_TTS_VOICE}`,
    `Dataset: ${job.dataset_ref || "-"}`,
    `Kernel: ${job.kernel_ref || "-"}`,
    "",
    ...(job.messages || []),
  ];
  if (targetDuration) {
    lines.splice(2, 0, `Thời lượng mục tiêu: ${targetDuration} giây (mềm)`);
  }
  if (plannedBacking) {
    lines.splice(targetDuration ? 3 : 2, 0, `Nhạc nền dự kiến: khoảng ${plannedBacking} giây`);
  }
  if (durationCeiling) {
    lines.splice(targetDuration || plannedBacking ? 4 : 2, 0, `Giới hạn mềm: ${durationCeiling} giây`);
  }
  if (outroTail) {
    lines.splice(targetDuration || plannedBacking || durationCeiling ? 5 : 2, 0, `Đuôi outro: khoảng ${outroTail} giây`);
  }
  if (job.mp3_path) {
    lines.push("", `MP3: ${job.mp3_path}`);
  }
  if (job.last_error) {
    lines.push("", `Chi tiết lỗi: ${job.last_error}`);
  }
  if (job.commands?.length && job.status === "needs_setup") {
    lines.push("", "Chạy các lệnh này sau khi cấu hình Kaggle API:", ...job.commands);
  }
  kaggleJobBox.textContent = lines.join("\n");
  renderAudio(job);
  renderWarning(job);
  renderLyrics(job);
  renderDownloads(job);
  drawWave(job.status);
}

async function retryTts(runId) {
  if (!runId) return;
  setGenerating(true, "Đang thử lại TTS...");
  statusPill.textContent = "Thử lại TTS";
  try {
    const response = await fetch("/api/kaggle/retry-tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, model: currentJob?.model || DEFAULT_CUSTOM_MUSIC_MODEL }),
    });
    const job = await response.json();
    if (!response.ok || job.error) {
      throw new Error(job.error || "Không thể gửi lại job TTS");
    }
    renderJob(job);
    if (!["needs_setup", "failed", "complete"].includes(job.status)) {
      pollKaggle(job.run_id, "Đang thử lại TTS...");
    } else {
      setGenerating(false);
    }
  } catch (error) {
    statusPill.textContent = "Lỗi";
    kaggleJobBox.textContent = error.message;
    setGenerating(false);
  }
}

async function pollKaggle(runId, label = "Đang tạo...") {
  const pollId = activePollId + 1;
  activePollId = pollId;
  setGenerating(true, label);
  for (let i = 0; i < 120; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 15000));
    if (pollId !== activePollId) return;
    try {
      const response = await fetch(`/api/kaggle/status?run_id=${encodeURIComponent(runId)}`);
      const job = await response.json();
      if (!response.ok || job.error) {
        setGenerating(false);
        return;
      }
      renderJob(job);
      if (job.status === "complete") {
        statusPill.textContent = "Hoàn tất";
        setGenerating(false);
        return;
      }
      if (job.status === "failed" || job.status === "needs_setup") {
        statusPill.textContent = job.status === "needs_setup" ? "Cần cấu hình Kaggle" : "Lỗi";
        setGenerating(false);
        return;
      }
    } catch (_error) {
      setGenerating(false);
      return;
    }
  }
  setGenerating(false);
}

function renderAudio(job) {
  if (!job.mp3_url) {
    audioSlot.textContent = "MP3 sẽ xuất hiện ở đây sau khi job Kaggle hoàn tất.";
    return;
  }
  audioSlot.innerHTML = `<audio controls src="${job.mp3_url}"></audio>`;
}

function renderWarning(job) {
  if (!warningSlot) return;
  if (!job) {
    warningSlot.hidden = true;
    warningSlot.textContent = "";
    return;
  }

  const backend = `${job.generation_backend || job.backend || ""}`;
  const hasVocalMix = backend.includes("f5_tts_vocal_mix") || backend.includes("mms_tts_vocal_mix");
  const warnings = [];
  const customFailed = Boolean(job.custom_model_failed || job.custom_model_error);
  const musicgenFailed = Boolean(job.musicgen_failed || job.musicgen_error || backend.includes("guide_fallback"));
  const ttsFailed = Boolean(
    job.vocal_failed || job.tts_error || backend.includes("tts_failed") || backend.includes("tts_skipped"),
  );
  const f5FallbackUsed = Boolean(job.f5_tts_error && backend.includes("f5_failed_mms"));
  if (customFailed) {
    const detail = summarizeError(job.custom_model_error || job.last_error || "");
    warnings.push([
      "Custom composer bị lỗi trên Kaggle. Chưa thể tạo backing track để mix vocal.",
      detail ? `Chi tiết: ${detail}` : "",
    ].filter(Boolean).join("\n"));
  }
  if (musicgenFailed) {
    const detail = summarizeError(job.musicgen_error || job.last_error || "");
    warnings.push([
      "Job cũ dùng MusicGen bị lỗi. MP3 hiện tại có thể chỉ là fallback backing.",
      detail ? `Chi tiết: ${detail}` : "",
    ].filter(Boolean).join("\n"));
  }
  if (f5FallbackUsed) {
    const detail = summarizeError(job.f5_tts_error || "");
    warnings.push([
      "F5-TTS bị lỗi trên Kaggle, hệ thống đã tự dùng MMS Vietnamese TTS fallback để vẫn có vocal.",
      detail ? `Chi tiết F5: ${detail}` : "",
    ].filter(Boolean).join("\n"));
  }
  if (ttsFailed) {
    const detail = summarizeError(job.tts_error || job.last_error || "");
    warnings.push([
      "TTS/Vocal bị lỗi. MP3 hiện tại chỉ là nhạc nền, chưa có giọng hát.",
      detail ? `Chi tiết: ${detail}` : "",
    ].filter(Boolean).join("\n"));
  }
  if (warnings.length) {
    warningSlot.hidden = false;
    warningSlot.textContent = warnings.join("\n\n");
    return;
  }

  if (job.mp3_url && job.lyrics_text && !job.vocal_url && backend && !hasVocalMix) {
    warningSlot.hidden = false;
    warningSlot.textContent = "Không nhận được file vocal WAV từ Kaggle. MP3 có thể chỉ là nhạc nền.";
    return;
  }

  warningSlot.hidden = true;
  warningSlot.textContent = "";
}

function summarizeError(value) {
  return String(value || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-4)
    .join(" ")
    .slice(0, 420);
}

function renderDownloads(job) {
  const links = [];
  if (job.mp3_url) {
    links.push(`<a href="${job.mp3_url}" download>Tải MP3</a>`);
  }
  if (job.lyrics_url) {
    links.push(`<a href="${job.lyrics_url}" download>Tải lời bài hát</a>`);
  }
  if (job.vocal_url) {
    links.push(`<a href="${job.vocal_url}" download>Tải vocal WAV</a>`);
  }
  if (job.backing_url) {
    links.push(`<a href="${job.backing_url}" download>Tải nhạc nền MP3</a>`);
  }
  if (canRetryTts(job)) {
    links.push(`<button type="button" class="download-action" data-retry-tts="${job.run_id}">Thử lại TTS</button>`);
  }
  if (!links.length) {
    downloads.innerHTML = "";
    return;
  }
  downloads.innerHTML = links.join("");
}

function canRetryTts(job) {
  if (!job || job.status !== "complete" || !job.mp3_url || job.vocal_url) return false;
  const backend = `${job.generation_backend || job.backend || ""}`;
  const hasVocalMix = backend.includes("f5_tts_vocal_mix") || backend.includes("mms_tts_vocal_mix");
  return Boolean(
    job.vocal_failed ||
      job.tts_error ||
      backend.includes("tts_failed") ||
      backend.includes("tts_skipped") ||
      (job.lyrics_text && !hasVocalMix),
  );
}

function renderLyrics(job) {
  const lines = [];
  const vocal = job.vocal_plan || {};
  const scene = job.scene_plan || job.analysis?.scene || {};
  if (Array.isArray(scene.labels) && scene.labels.length) {
    lines.push(`Cảnh: ${scene.labels.join(", ")}`);
    if (Array.isArray(scene.ambience_layers) && scene.ambience_layers.length) {
      lines.push(`Không khí: ${scene.ambience_layers.join(", ")}`);
    }
    if (Array.isArray(scene.prompt_cues) && scene.prompt_cues.length) {
      lines.push(`Gợi ý prompt: ${scene.prompt_cues.slice(0, 4).join(", ")}`);
    }
    lines.push("");
  }
  if (Object.keys(vocal).length) {
    lines.push(`Ca sĩ gợi ý: ${formatVocalGender(vocal.gender)} ${vocal.register || ""}`.trim());
    lines.push(`Giọng TTS thực tế: ${job.tts_voice_actual || DEFAULT_TTS_VOICE}`);
    if (job.tts_voice_note) lines.push(`Ghi chú TTS: ${job.tts_voice_note}`);
    lines.push(`Cao độ: ${vocal.pitch_center || "-"} | Âm vực: ${vocal.range_low || "-"}-${vocal.range_high || "-"}`);
    if (vocal.delivery) lines.push(`Cách hát: ${vocal.delivery}`);
    if (vocal.intensity) lines.push(`Cường độ: ${vocal.intensity}`);
    if (Array.isArray(vocal.rationale) && vocal.rationale.length) {
      lines.push(`Lý do: ${vocal.rationale.slice(0, 2).join(" ")}`);
    }
    lines.push("");
  }

  if (job.lyrics_text) {
    lines.push(job.lyrics_text);
  } else if (Array.isArray(job.lyrics?.full_song)) {
    lines.push(job.lyrics.full_song.join("\n"));
  }

  lyricsOutput.textContent = lines.join("\n").trim() || "Lời bài hát và kế hoạch vocal sẽ xuất hiện ở đây sau khi job Kaggle hoàn tất.";
}

function formatVocalGender(value) {
  if (value === "female") return "Nữ";
  if (value === "male") return "Nam";
  if (value === "duet") return "Song ca";
  return "Tự động";
}

function formatJobStatus(value) {
  const labels = {
    complete: "hoàn tất",
    failed: "lỗi",
    needs_setup: "cần cấu hình",
    submitting: "đang gửi",
    queued: "đang xếp hàng",
    running: "đang chạy",
  };
  return labels[value] || value || "chưa rõ";
}

function drawWave(status = "idle") {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7fbfa";
  ctx.fillRect(0, 0, width, height);

  const bars = 24;
  const barWidth = width / bars;
  const active = status !== "idle" && status !== "needs_setup";
  for (let i = 0; i < bars; i += 1) {
    const amp = active ? Math.sin(i * 1.1) * 0.24 + 0.5 : 0.18;
    const x = i * barWidth;
    const y = height * (0.5 - amp * 0.36);
    const h = height * amp * 0.72;
    ctx.fillStyle = i % 4 === 0 ? "#c7562c" : "#0f766e";
    ctx.fillRect(x + 3, y, Math.max(4, barWidth - 6), h);
  }
}

drawWave();
