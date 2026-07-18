const form = document.querySelector("#generate-form");
const duration = document.querySelector("#duration");
const durationValue = document.querySelector("#duration-value");
const statusPill = document.querySelector("#status-pill");
const generateButton = document.querySelector("#generate-button");
const buttonText = generateButton.querySelector(".button-text");
const jobBox = document.querySelector("#kaggle-job");
const kaggleLinks = document.querySelector("#kaggle-links");
const downloads = document.querySelector("#downloads");
const audioSlot = document.querySelector("#audio-slot");
const warningSlot = document.querySelector("#warning-slot");
const lyricsOutput = document.querySelector("#lyrics-output");
const canvas = document.querySelector("#wave-canvas");
const ctx = canvas.getContext("2d");
let isGenerating = false;
let activePollId = 0;
const MODEL = "genmusic-vn-self-diffusion-v1";

duration.addEventListener("input", () => {
  durationValue.textContent = duration.value + " giây";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isGenerating) return;
  setGenerating(true, "Đang gửi...");
  statusPill.textContent = "Đang gửi";
  downloads.innerHTML = "";
  warningSlot.hidden = true;
  audioSlot.textContent = "Đang chuẩn bị model trên Kaggle...";
  lyricsOutput.textContent = "Đang chuẩn bị LRC...";
  jobBox.textContent = "";
  kaggleLinks.innerHTML = "";
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        text: document.querySelector("#text").value,
        duration_seconds: Number(duration.value),
        genre: document.querySelector("#genre").value,
        model: MODEL,
      }),
    });
    const job = await response.json();
    if (!response.ok || job.error) throw new Error(job.error || "Không thể gửi job");
    renderJob(job);
    if (["staged", "submitted", "dataset_uploaded", "running"].includes(job.status)) pollKaggle(job.run_id);
    else setGenerating(false);
  } catch (error) {
    statusPill.textContent = "Lỗi";
    jobBox.textContent = error.message;
    audioSlot.textContent = "Không có audio vì request chưa được gửi.";
    lyricsOutput.textContent = "Không thể tạo LRC từ input hiện tại.";
    warningSlot.textContent = error.message;
    warningSlot.hidden = false;
    drawWave("failed");
    setGenerating(false);
  }
});

function setGenerating(active, label) {
  isGenerating = active;
  generateButton.disabled = active;
  generateButton.classList.toggle("is-loading", active);
  generateButton.setAttribute("aria-busy", active ? "true" : "false");
  buttonText.textContent = label || "Tạo bản nhạc";
}

function renderJob(job) {
  const lines = [
    "Trạng thái: " + formatStatus(job.status),
    "Backend: " + (job.backend || MODEL),
    "Model: " + (job.model || MODEL),
    "Thời lượng: " + (job.duration_seconds || "-") + " giây",
    "Checkpoint: " + (job.checkpoint_kernel_ref || "-"),
    "Kernel: " + (job.kernel_ref || "-"),
    "",
    ...(job.messages || []),
  ];
  if (job.last_error) lines.push("", "Lỗi: " + job.last_error);
  jobBox.textContent = lines.join("\n");
  lyricsOutput.textContent = typeof job.lyrics === "string" ? job.lyrics : JSON.stringify(job.lyrics || "LRC đã được tạo trong request.", null, 2);
  renderKaggleLinks(job);
  renderAudio(job);
  renderDownloads(job);
  drawWave(job.status);
  if (job.status === "needs_setup") statusPill.textContent = "Cần cấu hình Kaggle";
  if (job.status === "failed") statusPill.textContent = "Lỗi";
  if (job.status === "complete") statusPill.textContent = "Hoàn tất";
}

function renderKaggleLinks(job) {
  kaggleLinks.innerHTML = "";
  const links = [
    ["Mở checkpoint Kaggle", job.checkpoint_url],
    ["Mở kernel Kaggle", job.kernel_url],
  ];
  for (const [label, url] of links) {
    if (!url) continue;
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    anchor.textContent = label;
    kaggleLinks.appendChild(anchor);
  }
}

function renderAudio(job) {
  const audioUrl = job.mp3_url || job.wav_url;
  audioSlot.replaceChildren();
  if (!audioUrl) {
    audioSlot.textContent = "Audio sẽ xuất hiện sau khi job hoàn tất.";
    return;
  }
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = audioUrl;
  audioSlot.appendChild(audio);
}

function renderDownloads(job) {
  downloads.replaceChildren();
  const links = [
    ["Tải MP3", job.mp3_url],
    ["Tải WAV", job.wav_url],
    ["Tải LRC", job.lrc_url],
  ];
  for (const [label, url] of links) {
    if (!url) continue;
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "";
    anchor.textContent = label;
    downloads.appendChild(anchor);
  }
}

async function pollKaggle(runId) {
  const pollId = ++activePollId;
  let consecutiveErrors = 0;
  setGenerating(true, "Đang tạo và mix...");
  for (let index = 0; index < 120; index += 1) {
    await new Promise((resolve) => setTimeout(resolve, 15000));
    if (pollId !== activePollId) return;
    try {
      const response = await fetch("/api/kaggle/status?run_id=" + encodeURIComponent(runId));
      const job = await response.json();
      if (!response.ok || job.error) throw new Error(job.error || "Không đọc được trạng thái job");
      consecutiveErrors = 0;
      warningSlot.hidden = true;
      renderJob(job);
      if (["complete", "failed", "needs_setup"].includes(job.status)) {
        setGenerating(false);
        return;
      }
    } catch (error) {
      consecutiveErrors += 1;
      warningSlot.textContent = `Mất kết nối tạm thời (${consecutiveErrors}/3): ${error.message}`;
      warningSlot.hidden = false;
      statusPill.textContent = consecutiveErrors < 3 ? "Đang thử lại" : "Lỗi";
      if (consecutiveErrors >= 3) {
        jobBox.textContent = error.message;
        setGenerating(false);
        drawWave("failed");
        return;
      }
    }
  }
  setGenerating(false);
}

function formatStatus(value) {
  const labels = {complete: "hoàn tất", failed: "lỗi", needs_setup: "cần cấu hình Kaggle", staged: "đã stage", submitted: "đã submit", running: "đang chạy", dataset_uploaded: "đã tải dataset"};
  return labels[value] || value || "chưa rõ";
}

function drawWave(status) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f7fbfa";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const active = status && !["idle", "failed"].includes(status);
  for (let i = 0; i < 28; i += 1) {
    const amp = active ? 0.25 + (Math.sin(i * 1.3) + 1) * 0.2 : 0.18;
    const width = canvas.width / 28;
    const height = canvas.height * amp;
    ctx.fillStyle = i % 4 === 0 ? "#c7562c" : "#0f766e";
    ctx.fillRect(i * width + 3, canvas.height / 2 - height / 2, Math.max(4, width - 6), height);
  }
}

drawWave("idle");
