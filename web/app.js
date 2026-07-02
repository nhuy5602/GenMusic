const form = document.querySelector("#generate-form");
const duration = document.querySelector("#duration");
const durationValue = document.querySelector("#duration-value");
const statusPill = document.querySelector("#status-pill");
const emotion = document.querySelector("#emotion");
const chords = document.querySelector("#chords");
const lyrics = document.querySelector("#lyrics");
const promptBox = document.querySelector("#prompt");
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
  statusPill.textContent = "Dang tao";
  downloads.innerHTML = "";
  audioSlot.innerHTML = "";
  kaggleJobBox.textContent = "";

  const data = {
    text: document.querySelector("#text").value,
    duration_seconds: Number(duration.value),
    genre: document.querySelector("#genre").value,
    backend: document.querySelector("input[name='backend']:checked").value,
    kaggle_backend: "musicgen",
    model: "facebook/musicgen-small",
  };

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await response.json();
    if (!response.ok || result.error) {
      throw new Error(result.error || "Khong tao duoc output");
    }
    renderResult(result);
    statusPill.textContent = result.kaggle_job ? "Kaggle" : "Hoan tat";
    if (result.kaggle_job) pollKaggle(result.run_id);
  } catch (error) {
    statusPill.textContent = "Loi";
    promptBox.textContent = error.message;
  }
});

function renderResult(result) {
  emotion.innerHTML = [
    ["Mood", result.emotion.label_vi],
    ["Valence", result.emotion.valence],
    ["Energy", result.emotion.energy],
    ["Confidence", result.emotion.confidence],
    ["Keywords", result.emotion.keywords.join(", ")],
  ]
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");

  chords.innerHTML = result.harmony.chord_progression
    .map((chord) => `<div class="chord">${chord}</div>`)
    .join("");

  lyrics.textContent = [
    result.lyrics.title,
    "",
    "[Verse]",
    ...result.lyrics.verse,
    "",
    "[Chorus]",
    ...result.lyrics.chorus,
    "",
    "[Bridge]",
    ...result.lyrics.bridge,
  ].join("\n");

  promptBox.textContent = result.prompt;
  renderKaggleJob(result.kaggle_job);
  renderDownloads(result.files);
  renderAudio(result.files);
  drawWave(result);
}

function renderKaggleJob(job) {
  if (!job) {
    kaggleJobBox.textContent = "Chon Kaggle Auto de submit GPU job va sync output ve local.";
    return;
  }

  const lines = [
    `Status: ${job.status}`,
    `Dataset: ${job.dataset_ref}`,
    `Kernel: ${job.kernel_ref}`,
    "",
    ...(job.messages || []),
  ];
  if (job.downloaded_files?.length) {
    lines.push("", "Downloaded:", ...job.downloaded_files);
  }
  if (job.commands?.length && job.status === "needs_setup") {
    lines.push("", "Run after Kaggle API setup:", ...job.commands);
  }
  kaggleJobBox.textContent = lines.join("\n");
}

async function pollKaggle(runId) {
  for (let i = 0; i < 120; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 15000));
    try {
      const response = await fetch(`/api/kaggle/status?run_id=${encodeURIComponent(runId)}`);
      const job = await response.json();
      if (!response.ok || job.error) return;
      renderKaggleJob(job);
      if (job.status === "complete") {
        statusPill.textContent = "Hoan tat";
        return;
      }
      if (job.status === "failed" || job.status === "needs_setup") {
        statusPill.textContent = job.status === "needs_setup" ? "Can API" : "Loi";
        return;
      }
    } catch (_error) {
      return;
    }
  }
}

function renderDownloads(files) {
  downloads.innerHTML = files
    .filter((file) => file.url)
    .map((file) => `<a href="${file.url}" download>${file.kind}</a>`)
    .join("");
}

function renderAudio(files) {
  const audio = files.find((file) => file.kind === "audio" && file.url);
  if (!audio) {
    audioSlot.textContent = "Audio se xuat hien o day khi backend tao xong.";
    return;
  }
  audioSlot.innerHTML = `<audio controls src="${audio.url}"></audio>`;
}

function drawWave(result) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7fbfa";
  ctx.fillRect(0, 0, width, height);

  const bars = result.harmony.chord_progression.length * 4;
  const barWidth = width / bars;
  const energy = Math.max(0.1, Number(result.emotion.energy || 0.4));

  for (let i = 0; i < bars; i += 1) {
    const amp = (Math.sin(i * 1.7) * 0.5 + 0.5) * energy;
    const x = i * barWidth;
    const y = height * (0.5 - amp * 0.34);
    const h = height * amp * 0.68;
    ctx.fillStyle = i % 4 === 0 ? "#c7562c" : "#0f766e";
    ctx.fillRect(x + 3, y, Math.max(4, barWidth - 6), h);
  }

  ctx.strokeStyle = "#202124";
  ctx.lineWidth = 2;
  ctx.beginPath();
  result.melody.forEach((note, index) => {
    const x = (note.start / result.duration_seconds) * width;
    const y = height - ((note.midi - 48) / 36) * height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

drawWave({
  duration_seconds: 24,
  emotion: { energy: 0.45 },
  harmony: { chord_progression: ["Am", "F", "C", "G"] },
  melody: [],
});

