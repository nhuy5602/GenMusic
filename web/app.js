const form = document.querySelector("#generate-form");
const duration = document.querySelector("#duration");
const durationValue = document.querySelector("#duration-value");
const statusPill = document.querySelector("#status-pill");
const kaggleJobBox = document.querySelector("#kaggle-job");
const downloads = document.querySelector("#downloads");
const audioSlot = document.querySelector("#audio-slot");
const canvas = document.querySelector("#wave-canvas");
const ctx = canvas.getContext("2d");

duration.addEventListener("input", () => {
  durationValue.textContent = `${duration.value}s`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusPill.textContent = "Submitting";
  downloads.innerHTML = "";
  audioSlot.textContent = "Waiting for Kaggle MusicGen output...";
  kaggleJobBox.textContent = "";

  const data = {
    text: document.querySelector("#text").value,
    duration_seconds: Number(duration.value),
    genre: document.querySelector("#genre").value,
    model: "facebook/musicgen-small",
  };

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const job = await response.json();
    if (!response.ok || job.error) {
      throw new Error(job.error || "Could not submit job");
    }
    renderJob(job);
    statusPill.textContent = job.status === "needs_setup" ? "Needs API" : "Kaggle";
    if (!["needs_setup", "failed", "complete"].includes(job.status)) {
      pollKaggle(job.run_id);
    }
  } catch (error) {
    statusPill.textContent = "Error";
    kaggleJobBox.textContent = error.message;
  }
});

function renderJob(job) {
  const lines = [
    `Status: ${job.status}`,
    `Model: ${job.model}`,
    `Dataset: ${job.dataset_ref}`,
    `Kernel: ${job.kernel_ref}`,
    "",
    ...(job.messages || []),
  ];
  if (job.mp3_path) {
    lines.push("", `MP3: ${job.mp3_path}`);
  }
  if (job.commands?.length && job.status === "needs_setup") {
    lines.push("", "Run after Kaggle API setup:", ...job.commands);
  }
  kaggleJobBox.textContent = lines.join("\n");
  renderAudio(job);
  renderDownloads(job);
  drawWave(job.status);
}

async function pollKaggle(runId) {
  for (let i = 0; i < 120; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 15000));
    try {
      const response = await fetch(`/api/kaggle/status?run_id=${encodeURIComponent(runId)}`);
      const job = await response.json();
      if (!response.ok || job.error) return;
      renderJob(job);
      if (job.status === "complete") {
        statusPill.textContent = "Done";
        return;
      }
      if (job.status === "failed" || job.status === "needs_setup") {
        statusPill.textContent = job.status === "needs_setup" ? "Needs API" : "Error";
        return;
      }
    } catch (_error) {
      return;
    }
  }
}

function renderAudio(job) {
  if (!job.mp3_url) {
    audioSlot.textContent = "MP3 will appear here after the Kaggle job finishes.";
    return;
  }
  audioSlot.innerHTML = `<audio controls src="${job.mp3_url}"></audio>`;
}

function renderDownloads(job) {
  if (!job.mp3_url) {
    downloads.innerHTML = "";
    return;
  }
  downloads.innerHTML = `<a href="${job.mp3_url}" download>Download MP3</a>`;
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

