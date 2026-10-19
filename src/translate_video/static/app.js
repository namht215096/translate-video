"use strict";

const $ = id => document.getElementById(id);
let context, originalGain, dubGain, sourceNode, decoded, inputPCM;
let socket, session = 0, running = false, paused = false, previewing = false, ended = false;
let cursor = 0, sourceStart = 0, previewStart = 0, timer, startupTimer;
let inputProcessed = 0, inputEnded = false, serverDone = false;
const LOOKAHEAD_SECONDS = 5, MAX_INPUT_IN_FLIGHT = 64000;
let nextDubTime = 0, sampleRate = 48000, dubNodes = new Set(), drainId = null;
let queueTimes = [], translationTimes = [], speechTimes = [], rows = new Map();
let wavePosition = 0;
let requestedAt = 0, firstSpeechDelayMs = null;

function status(text, error = false) {
  $("statusMessage").textContent = text;
  $("statusMessage").classList.toggle("error", error);
}
function tag(text, live = false) {
  $("liveTag").textContent = text;
  $("liveTag").classList.toggle("live", live);
}
function clock(seconds) {
  seconds = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
function send(value) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(value));
}
function audioContext() {
  if (!context) {
    context = new AudioContext();
    originalGain = context.createGain(); originalGain.connect(context.destination);
    dubGain = context.createGain(); dubGain.connect(context.destination);
    updateVolume();
  }
  return context;
}
function updateVolume() {
  $("originalValue").textContent = `${$("originalVolume").value}%`;
  $("dubValue").textContent = `${$("dubVolume").value}%`;
  if (context) {
    originalGain.gain.value = Number($("originalVolume").value) / 100;
    dubGain.gain.value = Number($("dubVolume").value) / 100;
  }
}
function buttons() {
  $("startButton").disabled = !decoded || (running && !sourceNode);
  $("startButton").innerHTML = running ? (!sourceNode ? "Đang chuẩn bị…" : paused ? "▶ Tiếp tục" : "Ⅱ Tạm dừng") : "▶ Phát & dịch";
  $("previewButton").disabled = !decoded || running;
  $("previewButton").textContent = previewing ? "Dừng nghe thử" : "Nghe bản gốc";
  $("stopButton").disabled = !running && !previewing;
  $("changeFile").disabled = running;
  $("dropZone").disabled = running;
  $("sampleButton").disabled = running;
}
function resetMetrics() {
  queueTimes = []; translationTimes = []; speechTimes = []; rows.clear();
  for (const id of ["queueP95", "translationP95", "speechP95"]) $(id).textContent = "—";
  $("sampleCount").textContent = "Chưa có lượt đo";
  $("segmentCount").textContent = "0 đoạn";
  $("transcript").replaceChildren();
}
function p95(values) {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.ceil(ordered.length * .95) - 1];
}
function latency(values) {
  if (!values.length) return "—";
  const value = p95(values);
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}
function metrics() {
  $("queueP95").textContent = latency(queueTimes);
  $("translationP95").textContent = latency(translationTimes);
  $("speechP95").textContent = latency(speechTimes);
  $("sampleCount").textContent = `${speechTimes.length} đoạn đo TTS · ít mẫu thì P95 chưa ổn định`;
}
function drawWave(position = wavePosition) {
  wavePosition = position;
  if (!decoded) return;
  const canvas = $("waveform");
  const width = canvas.clientWidth || 480;
  canvas.width = width * devicePixelRatio; canvas.height = 110 * devicePixelRatio;
  const ctx = canvas.getContext("2d"); ctx.scale(devicePixelRatio, devicePixelRatio);
  const data = decoded.getChannelData(0), bars = Math.floor(width / 5);
  for (let i = 0; i < bars; i++) {
    const start = Math.floor(i * data.length / bars), stop = Math.floor((i + 1) * data.length / bars);
    let peak = 0;
    for (let j = start; j < stop; j += Math.max(1, Math.floor((stop - start) / 80))) peak = Math.max(peak, Math.abs(data[j]));
    const height = Math.max(5, Math.min(92, peak * 100));
    ctx.fillStyle = i / bars <= position ? "#608861" : "#dbe6d5";
    ctx.beginPath(); ctx.roundRect(i * 5, (110 - height) / 2, 3, height, 2); ctx.fill();
  }
}
async function chooseFile(file) {
  if (!file || running) return;
  stopAll();
  const generation = session;
  if (file.size > 30 * 1024 * 1024) { status("File quá lớn. Hãy chọn file dưới 30 MB.", true); return; }
  status("Đang đọc file âm thanh…");
  $("startButton").disabled = true;
  try {
    const audio = await audioContext().decodeAudioData(await file.arrayBuffer());
    if (generation !== session) return;
    if (!audio.duration || audio.duration > 600) throw new Error("Hãy chọn audio không quá 10 phút.");
    const offline = new OfflineAudioContext(1, Math.ceil(audio.duration * 16000), 16000);
    const node = offline.createBufferSource(); node.buffer = audio; node.connect(offline.destination); node.start();
    const mono = await offline.startRendering();
    if (generation !== session) return;
    decoded = audio; inputPCM = mono.getChannelData(0);
    // Lossy decoding/resampling can overshoot even when the recording is valid.
    for (let i = 0; i < inputPCM.length; i++) inputPCM[i] = Math.max(-1, Math.min(1, inputPCM[i]));
    $("fileName").textContent = file.name;
    $("fileInfo").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ${clock(audio.duration)} · Đã sẵn sàng`;
    $("dropZone").hidden = true; $("sampleRow").hidden = true; $("filePanel").hidden = false;
    $("totalTime").textContent = clock(audio.duration); $("currentTime").textContent = "00:00";
    $("partialText").textContent = "Bấm Phát & dịch để bắt đầu nghe.";
    $("currentTranslation").textContent = "Bản dịch tiếng Việt sẽ xuất hiện tại đây.";
    tag("SẴN SÀNG"); drawWave(0);
    status("Đã đọc file. Bấm Phát & dịch, hoặc Nghe bản gốc để nghe thử.");
  } catch (error) {
    if (generation !== session) return;
    decoded = null; inputPCM = null;
    $("dropZone").hidden = false; $("sampleRow").hidden = false; $("filePanel").hidden = true;
    status(`Không đọc được audio: ${error.message} Hãy thử MP3 hoặc WAV khác.`, true);
  }
  buttons();
}
function stopSource() {
  if (sourceNode) { sourceNode.onended = null; try { sourceNode.stop(); } catch (_) {} sourceNode.disconnect(); sourceNode = null; }
}
function stopAll() {
  session++; running = paused = previewing = false; ended = false;
  clearInterval(timer); clearTimeout(startupTimer); stopSource();
  inputProcessed = 0; inputEnded = false; serverDone = false;
  for (const node of dubNodes) { node.onended = null; try { node.stop(); } catch (_) {} node.disconnect(); }
  dubNodes.clear(); drainId = null; nextDubTime = 0;
  if (socket) { socket.onclose = null; socket.close(); socket = null; }
  if (context?.state === "suspended") context.resume().catch(() => {});
  $("speakingBars").classList.remove("active");
  $("currentTime").textContent = "00:00"; drawWave(0); buttons();
}
function finished() {
  running = paused = false; clearInterval(timer); stopSource();
  $("currentTime").textContent = clock(decoded.duration); drawWave(1);
  tag("HOÀN THÀNH"); status("Đã phát và dịch xong. Bạn có thể nghe lại hoặc chọn file khác.");
  $("speakingBars").classList.remove("active"); buttons();
}
function startSource(at = context.currentTime) {
  if (!running || sourceNode) return;
  clearTimeout(startupTimer);
  sourceNode = context.createBufferSource(); sourceNode.buffer = decoded; sourceNode.connect(originalGain);
  sourceStart = at; sourceNode.start(at);
  sourceNode.onended = () => {
    ended = true;
    if (serverDone) finished();
    else if (running) { pumpInput(); tag("ĐANG ĐỌC NỐT", true); status("Bản gốc đã hết. Đang đọc nốt các đoạn dịch…"); }
  };
  buttons();
}
function primePlayback() {
  tag("CHUẨN BỊ TIẾNG VIỆT", true);
  status("Đang đọc trước một đoạn ngắn. Hai tiếng sẽ bắt đầu khi có audio tiếng Việt…");
  timer = setInterval(pumpInput, 40);
  pumpInput();
  // Silence / a very slow model must not leave the original stuck forever.
  startupTimer = setTimeout(() => {
    if (running && !sourceNode) { startSource(); tag("ĐANG DỊCH", true); status("Đang phát bản gốc; tiếng Việt sẽ vào khi sẵn sàng."); }
  }, 2000);
}
function pumpInput() {
  if (!running || paused) return;
  const elapsed = sourceNode ? Math.min(decoded.duration, Math.max(0, context.currentTime - sourceStart)) : 0;
  $("currentTime").textContent = clock(elapsed); drawWave(elapsed / decoded.duration);
  if (serverDone || socket?.readyState !== WebSocket.OPEN || inputEnded) return;
  const target = Math.min(inputPCM.length, Math.floor((elapsed + LOOKAHEAD_SECONDS) * 16000));
  if (socket.bufferedAmount > 256 * 1024) { fail("Kết nối không theo kịp audio. Hãy thử lại."); return; }
  while (cursor < inputPCM.length && cursor - inputProcessed < MAX_INPUT_IN_FLIGHT) {
    const next = Math.min(cursor + 1600, inputPCM.length);
    if (next > target) break;
    socket.send(inputPCM.slice(cursor, next).buffer); cursor = next;
  }
  if (cursor === inputPCM.length && !inputEnded) {
    inputEnded = true; send({type: "end"});
  }
}
function acknowledgeWhenBuffered(id, generation, connection) {
  if (generation !== session || connection !== socket || !running) return;
  if (!paused && nextDubTime - context.currentTime <= .8) send({type: "ack", id});
  else setTimeout(() => acknowledgeWhenBuffered(id, generation, connection), 35);
}
function acknowledgeDrain() {
  if (drainId !== null && dubNodes.size === 0 && !paused) {
    send({type: "ack", id: drainId}); drainId = null;
    $("speakingBars").classList.remove("active");
  }
}
function playChunk(bytes, generation, connection) {
  if (generation !== session || !running) return;
  const id = new DataView(bytes).getUint32(0, true);
  const pcm = new Float32Array(bytes.slice(4));
  const buffer = context.createBuffer(1, pcm.length, sampleRate); buffer.copyToChannel(pcm, 0);
  const node = context.createBufferSource(); node.buffer = buffer; node.connect(dubGain);
  nextDubTime = Math.max(context.currentTime + .035, nextDubTime);
  if (!sourceNode && !ended) startSource(nextDubTime);
  if (firstSpeechDelayMs === null) {
    firstSpeechDelayMs = performance.now() - requestedAt + (nextDubTime - context.currentTime) * 1000;
    status(`Tiếng Việt bắt đầu sau ${(firstSpeechDelayMs / 1000).toFixed(1)} giây · Có đọc trước MP3.`);
  }
  node.start(nextDubTime); nextDubTime += buffer.duration; dubNodes.add(node);
  $("speakingBars").classList.add("active");
  node.onended = () => { dubNodes.delete(node); node.disconnect(); acknowledgeDrain(); };
  acknowledgeWhenBuffered(id, generation, connection);
}
function showTranslation(event) {
  $("partialText").textContent = event.source;
  $("currentTranslation").textContent = event.translation;
  if (!rows.has(event.index)) {
    const row = document.createElement("div"); row.className = "transcript-row";
    for (const text of [event.source, event.translation]) { const p = document.createElement("p"); p.textContent = text; row.appendChild(p); }
    $("transcript").appendChild(row); rows.set(event.index, row);
    $("transcript").scrollTop = $("transcript").scrollHeight;
    $("segmentCount").textContent = `${rows.size} đoạn`;
    queueTimes.push(event.translation_queue_ms); translationTimes.push(event.translation_ms); metrics();
  }
}
function fail(message) {
  stopAll(); tag("ĐÃ DỪNG"); status(message, true);
}
async function startOrPause() {
  if (!decoded) return;
  if (running) {
    if (!sourceNode) return;
    paused = !paused;
    if (paused) { send({type: "pause"}); await context.suspend(); tag("TẠM DỪNG"); status("Đã tạm dừng cả tiếng gốc và tiếng dịch."); }
    else { await context.resume(); send({type: "resume"}); acknowledgeDrain(); tag(serverDone ? "ĐANG PHÁT" : ended ? "ĐANG ĐỌC NỐT" : "ĐANG DỊCH", true); status("Đang phát và dịch…"); }
    buttons(); return;
  }
  stopAll(); resetMetrics(); await audioContext().resume();
  requestedAt = performance.now(); firstSpeechDelayMs = null;
  running = true; cursor = 0; ended = false; const generation = session;
  buttons(); tag("ĐANG CHUẨN BỊ"); status("Đang kết nối máy dịch…");
  // A cancelled GPU call may take a moment to finish. Never overlap sessions.
  const deadline = Date.now() + 120000;
  while (true) {
    const response = await fetch("/api/health", {cache: "no-store"});
    if (!response.ok) throw new Error("Không kết nối được máy dịch.");
    const info = await response.json();
    if (generation !== session || !running) return;
    if (info.load_error) throw new Error(info.load_error);
    if (!info.busy) break;
    if (Date.now() > deadline) throw new Error("Máy dịch vẫn đang bận. Hãy đóng phiên khác rồi thử lại.");
    status(info.loading ? "Đang nạp và làm nóng model… Sau bước này có thể bắt đầu phát." : "Đang chờ phiên trước kết thúc xử lý… Có thể bấm Dừng để hủy.");
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  const connection = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`);
  socket = connection; connection.binaryType = "arraybuffer";
  connection.onmessage = message => {
    if (generation !== session) return;
    if (message.data instanceof ArrayBuffer) { playChunk(message.data, generation, connection); return; }
    const event = JSON.parse(message.data);
    if (event.type === "status") status(event.message);
    else if (event.type === "ready") { sampleRate = event.sample_rate; primePlayback(); buttons(); }
    else if (event.type === "input_progress") { inputProcessed = Math.max(inputProcessed, Math.round(event.seconds * 16000)); pumpInput(); }
    else if (event.type === "partial") $("partialText").textContent = event.text || "Đang nghe…";
    else if (event.type === "translation") showTranslation(event);
    else if (event.type === "audio_start") sampleRate = event.sample_rate;
    else if (event.type === "audio_end") { drainId = event.id; acknowledgeDrain(); }
    else if (event.type === "segment") { speechTimes.push(event.tts_first_chunk_ms); metrics(); }
    else if (event.type === "done") {
      serverDone = true;
      if (!sourceNode) startSource();
      if (ended) finished();
      else { tag("ĐANG PHÁT", true); status("Tiếng Việt đã phát xong. Đang phát nốt bản gốc…"); }
    }
    else if (event.type === "error") fail(event.message);
  };
  connection.onerror = () => { if (generation === session && running) fail("Không kết nối được máy dịch. Kiểm tra server Python đang chạy."); };
  connection.onclose = () => { if (generation === session && running && !serverDone) fail("Kết nối đã đóng. Bấm Phát & dịch để thử lại."); };
}
async function preview() {
  if (!decoded || running) return;
  if (previewing) { stopAll(); status("Đã dừng nghe thử."); return; }
  stopAll(); await audioContext().resume(); previewing = true;
  sourceNode = context.createBufferSource(); sourceNode.buffer = decoded;
  // Preview uses the same original-volume slider as live playback.
  sourceNode.connect(originalGain); previewStart = context.currentTime; sourceNode.start();
  sourceNode.onended = () => { stopAll(); tag("SẴN SÀNG"); status("Đã nghe xong bản gốc. Bấm Phát & dịch để nghe tiếng Việt."); };
  timer = setInterval(() => { const t = context.currentTime - previewStart; $("currentTime").textContent = clock(t); drawWave(t / decoded.duration); }, 80);
  status("Đang nghe thử bản gốc. Chưa chạy dịch."); buttons();
}

$("dropZone").onclick = () => $("fileInput").click();
$("changeFile").onclick = () => $("fileInput").click();
$("fileInput").onchange = event => { chooseFile(event.target.files[0]); event.target.value = ""; };
for (const name of ["dragenter", "dragover"]) $("dropZone").addEventListener(name, event => { event.preventDefault(); $("dropZone").classList.add("dragging"); });
for (const name of ["dragleave", "drop"]) $("dropZone").addEventListener(name, event => { event.preventDefault(); $("dropZone").classList.remove("dragging"); });
$("dropZone").addEventListener("drop", event => chooseFile(event.dataTransfer.files[0]));
document.addEventListener("dragover", event => event.preventDefault());
document.addEventListener("drop", event => { event.preventDefault(); if (!$("dropZone").contains(event.target)) chooseFile(event.dataTransfer.files[0]); });
$("startButton").onclick = () => startOrPause().catch(error => fail(error.message));
$("previewButton").onclick = () => preview().catch(error => fail(error.message));
$("stopButton").onclick = () => { stopAll(); tag("ĐÃ DỪNG"); status("Đã dừng. Lần phát tiếp theo sẽ bắt đầu từ đầu file."); };
for (const id of ["originalVolume", "dubVolume"]) $(id).oninput = updateVolume;
$("sampleButton").onclick = async () => {
  $("sampleButton").disabled = true;
  try { const response = await fetch("/api/example"); if (!response.ok) throw new Error("Không tìm thấy file mẫu. Hãy kéo MP3 từ máy vào."); await chooseFile(new File([await response.blob()], "english_10s.wav", {type: "audio/wav"})); }
  catch (error) { status(error.message, true); } finally { $("sampleButton").disabled = false; }
};
async function health() {
  try {
    const response = await fetch("/api/health", {cache: "no-store"}); if (!response.ok) throw new Error();
    const info = await response.json();
    $("serverStatus").classList.remove("offline");
    $("serverStatus").replaceChildren(document.createElement("i"), document.createTextNode(info.loading ? "Đang nạp model…" : info.load_error ? "Lỗi nạp model" : info.busy ? "Máy dịch đang hoạt động" : info.models_loaded ? "Model đã sẵn sàng" : "Máy dịch đã kết nối"));
  } catch (_) { $("serverStatus").classList.add("offline"); $("serverStatus").textContent = "Chưa kết nối máy dịch"; }
}
window.addEventListener("resize", () => drawWave());
window.addEventListener("beforeunload", () => { if (socket) socket.close(); });
health(); setInterval(health, 6000); buttons();
