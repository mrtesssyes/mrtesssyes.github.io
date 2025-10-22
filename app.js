// SSTV Web Emulator - simplified educational encoder/decoder
// Encoder: converts an image to a sequence of tones (simulated SSTV-like)
// Decoder: basic energy-based decoding from uploaded audio (very simple)

let audioCtx = null;
let generatedBuffer = null;
let masterGain = null;
let liveDecodeNode = null;
let liveDecodeAnalyser = null;
let liveDecodeAnimation = null;
let playingSource = null;
let playingStartTime = 0;
let playingOffsetSeconds = 0;
let playingPaused = false;

function ensureAudioCtx(){
  if(!audioCtx){
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = audioCtx.createGain();
    masterGain.gain.value = 1;
    masterGain.connect(audioCtx.destination);
  }
}

function fileToImage(file){
  return new Promise((resolve,reject)=>{
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = ()=>{ URL.revokeObjectURL(url); resolve(img); };
    img.onerror = (e)=>{ URL.revokeObjectURL(url); reject(e); };
    img.src = url;
  });
}

function drawPreview(img){
  const canvas = document.getElementById('previewCanvas');
  const ctx = canvas.getContext('2d');
  // fit into canvas
  const ratio = Math.min(canvas.width/img.width, canvas.height/img.height);
  const w = img.width*ratio, h=img.height*ratio;
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0,w,h);
}

function imageToGrayscaleSamples(img, cols=320, rows=256){
  // draw to offscreen canvas scaled to cols x rows
  const c = document.createElement('canvas'); c.width=cols; c.height=rows;
  const ctx = c.getContext('2d');
  ctx.drawImage(img,0,0,cols,rows);
  const data = ctx.getImageData(0,0,cols,rows).data;
  const samples = [];
  for(let y=0;y<rows;y++){
    const row = new Array(cols);
    for(let x=0;x<cols;x++){
      const i=(y*cols+x)*4;
      const r=data[i]/255, g=data[i+1]/255, b=data[i+2]/255;
      const gray = (0.299*r+0.587*g+0.114*b);
      row[x]=gray; // default grayscale; caller may expect Array for RGB mode
    }
    samples.push(row);
  }
  return samples; // rows array of Float32Array
}

function imageToRGBSamples(img, cols=320, rows=256){
  const c = document.createElement('canvas'); c.width=cols; c.height=rows;
  const ctx = c.getContext('2d');
  ctx.drawImage(img,0,0,cols,rows);
  const data = ctx.getImageData(0,0,cols,rows).data;
  const samples = [];
  for(let y=0;y<rows;y++){
    const row = new Array(cols);
    for(let x=0;x<cols;x++){
      const i=(y*cols+x)*4;
      const r=data[i]/255, g=data[i+1]/255, b=data[i+2]/255;
      row[x]=[r,g,b];
    }
    samples.push(row);
  }
  return samples;
}

// Helper: create a buffer with given float data
function createAudioBufferFromFloat(floatArr, sampleRate){
  return {buffer:floatArr, sampleRate};
}

// Implement a simplified Scottie S1-like encoder timing with sync pulses.
// We'll still use grayscale-to-frequency mapping for pixels, but adjust per-line timing.
function synthesizeSSTV(samples, sampleRate=44100, options={}){
  // samples: array rows x cols. options.durationSeconds: target total seconds.
  const rows = samples.length;
  const cols = samples[0].length;
  const totalPixels = rows * cols;
  // detect whether samples contain arrays (RGB) and set tone multiplier accordingly
  const sampleCell = samples[0] && samples[0][0];
  const toneMultiplier = Array.isArray(sampleCell) ? sampleCell.length : 1;
  const totalTones = totalPixels * toneMultiplier;

  // Scottie S1 (very simplified): each line has a VIS/sync period + pixel periods.
  // For our emulator we'll compute tone ms per pixel so full image fits duration.
  let tonePerPixelMs = 10;
  if(options.durationSeconds && options.durationSeconds > 0){
    const totalMs = options.durationSeconds * 1000;
    // Reserve a small fraction for syncs overhead (~2%):
    const effectiveMs = Math.max(100, totalMs * 0.98);
    // compute per-tone ms using totalTones (accounts for RGB channels)
    tonePerPixelMs = effectiveMs / totalTones;
    tonePerPixelMs = Math.max(1, Math.min(200, tonePerPixelMs));
  }

  const samplePerTone = Math.max(1, Math.floor(sampleRate * (tonePerPixelMs/1000)));

  // Precompute sizes
  const samplesPerLine = cols * samplePerTone * toneMultiplier;
  const syncPerLineSamples = Math.floor(sampleRate * 0.005); // 5ms sync tone approx
  const totalSamples = rows * (syncPerLineSamples + samplesPerLine);
  // We'll prepend a simple VIS-like header: a leader tone (1900Hz for 300ms), then a VIS tone sequence
  const leaderMs = 300;
  const leaderSamples = Math.floor(sampleRate*(leaderMs/1000));
  const visTones = [1900, 1200, 1900]; // small pattern
  const visSamples = visTones.length * Math.floor(sampleRate*0.05);
  const out = new Float32Array(leaderSamples + visSamples + totalSamples);
  let widx = 0;
  // leader
  for(let i=0;i<leaderSamples;i++) out[widx++] = Math.sin(2*Math.PI*1900*(i/sampleRate)) * 0.8;
  // vis pattern
  for(const vt of visTones){
    const vs = Math.floor(sampleRate*0.05);
    for(let i=0;i<vs;i++) out[widx++] = Math.sin(2*Math.PI*vt*(i/sampleRate)) * 0.8;
  }
  // now write lines

  for(let r=0;r<rows;r++){
    // simple sync tone (fixed 1200Hz)
    for(let i=0;i<syncPerLineSamples;i++){
      out[widx++] = Math.sin(2*Math.PI*1200*(i/sampleRate)) * 0.8;
    }
    // pixels
    const rowArr = samples[r];
    // If color mode is RGB, rows contain interleaved arrays [R,G,B] per pixel (handled by caller)
    for(let c=0;c<cols;c++){
      const cell = rowArr[c];
      if(Array.isArray(cell)){
        // cell = [r,g,b] each in 0..1; encode sequentially R,G,B
        const chans = [ cell[0], cell[1], cell[2] ];
        for(const bright of chans){
          const freq = 1500 + bright*(2300-1500);
          for(let s=0;s<samplePerTone;s++){
            const t = s / sampleRate;
            out[widx++] = Math.sin(2*Math.PI*freq*t) * 0.9;
          }
        }
      } else {
        const bright = rowArr[c];
        const freq = 1500 + bright*(2300-1500);
        for(let s=0;s<samplePerTone;s++){
          const t = s / sampleRate;
          out[widx++] = Math.sin(2*Math.PI*freq*t) * 0.9;
        }
      }
    }
  }
  const meta = { samplePerTone, tonePerPixelMs, toneMultiplier, cols, rows, totalTones: totalTones };
  meta.headerOffset = leaderSamples + visSamples;
  return { buffer: out, sampleRate, meta };
}

function playFloat32Buffer(bufObj){
  ensureAudioCtx();
  const {buffer,sampleRate} = bufObj;
  const audioBuffer = audioCtx.createBuffer(1, buffer.length, sampleRate);
  audioBuffer.copyToChannel(buffer,0,0);
  const src = audioCtx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(masterGain);
  const offset = playingOffsetSeconds || 0;
  src.start(0, offset);
  playingSource = src;
  playingStartTime = audioCtx.currentTime - offset;
  // clear any analyser-based live decode to avoid double drawing
  if(liveDecodeAnimation){ stopLiveDecoding(); }
  return src;
}

function floatToWav(float32Array, sampleRate){
  const buffer = new ArrayBuffer(44 + float32Array.length * 2);
  const view = new DataView(buffer);
  function writeString(view, offset, string){
    for(let i=0;i<string.length;i++) view.setUint8(offset+i,string.charCodeAt(i));
  }
  writeString(view,0,'RIFF');
  view.setUint32(4,36 + float32Array.length*2,true);
  writeString(view,8,'WAVE');
  writeString(view,12,'fmt ');
  view.setUint32(16,16,true);
  view.setUint16(20,1,true);
  view.setUint16(22,1,true);
  view.setUint32(24,sampleRate,true);
  view.setUint32(28,sampleRate*2,true);
  view.setUint16(32,2,true);
  view.setUint16(34,16,true);
  writeString(view,36,'data');
  view.setUint32(40,float32Array.length*2,true);
  // PCM 16-bit
  let offset=44;
  for(let i=0;i<float32Array.length;i++){
    let s = Math.max(-1,Math.min(1,float32Array[i]));
    view.setInt16(offset, s<0 ? s*0x8000 : s*0x7FFF, true);
    offset+=2;
  }
  return new Blob([view],{type:'audio/wav'});
}

// Goertzel-based decoder: detect strong frequency components corresponding to tone range and map to brightness.
// We'll scan each tone-frame and compute energy per candidate freq then pick the best.
function goertzel(samples, sampleRate, targetFreq){
  const N = samples.length;
  const k = Math.round((N * targetFreq) / sampleRate);
  const omega = (2*Math.PI*k)/N;
  const coeff = 2*Math.cos(omega);
  let q0=0, q1=0, q2=0;
  for(let i=0;i<N;i++){
    q0 = coeff * q1 - q2 + samples[i];
    q2 = q1; q1 = q0;
  }
  const real = q1 - q2 * Math.cos(omega);
  const imag = q2 * Math.sin(omega);
  return Math.sqrt(real*real + imag*imag);
}

// Improved decoder with sync/VIS detection, auto RGB detection and resolution control
async function decodeAudioBlob(blob, cols=320, rows=256, options={}){
  const array = await blob.arrayBuffer();
  ensureAudioCtx();
  const decoded = await audioCtx.decodeAudioData(array.slice(0));
  const chan = decoded.getChannelData(0);
  const sampleRate = decoded.sampleRate;

  // Get desired output width from UI resolution slider if present
  const desiredCols = parseInt(document.getElementById('resolutionSlider')?.value) || cols;

  // Step 1: sliding Goertzel at 1200Hz to find sync-like pulses
  const windowMs = 12; const windowSamples = Math.max(32, Math.floor(sampleRate * (windowMs/1000)));
  const hop = Math.floor(windowSamples/2);
  const mags = [];
  for(let off=0; off+windowSamples < chan.length; off+=hop){
    const frame = chan.subarray(off, off+windowSamples);
    const mag = goertzel(frame, sampleRate, 1200);
    mags.push({off, mag});
  }
  // find local maxima in mags
  const peaks = [];
  for(let i=1;i<mags.length-1;i++){
    if(mags[i].mag > mags[i-1].mag && mags[i].mag > mags[i+1].mag) peaks.push(mags[i]);
  }
  if(peaks.length === 0){
    // fallback to simple decode
    return simpleDecodeFallback(chan, sampleRate, desiredCols, rows, options);
  }
  // threshold peaks to the largest ones
  const magsOnly = peaks.map(p=>p.mag).slice().sort((a,b)=>a-b);
  const medianMag = magsOnly[Math.floor(magsOnly.length/2)] || 0;
  const strongPeaks = peaks.filter(p=>p.mag > Math.max(1, medianMag*3)).map(p=>p.off).sort((a,b)=>a-b);
  if(strongPeaks.length < 2){
    return simpleDecodeFallback(chan, sampleRate, desiredCols, rows, options);
  }
  // estimate line spacing (samples between sync pulses)
  const diffs = [];
  for(let i=1;i<strongPeaks.length;i++) diffs.push(strongPeaks[i]-strongPeaks[i-1]);
  diffs.sort((a,b)=>a-b);
  const medianDiff = diffs[Math.floor(diffs.length/2)] || diffs[0] || Math.floor(sampleRate*0.2);
  const samplesPerLine = Math.max(1, Math.round(medianDiff));

  // estimate samplesPerTone by autocorrelation within first detected line region
  const firstSync = strongPeaks[0];
  const nextSync = strongPeaks[1] || firstSync + samplesPerLine;
  const lineWindow = chan.subarray(firstSync, Math.min(nextSync, chan.length));
  // autocorrelation-like search for period
  let bestLag=0, bestScore=Infinity;
  const maxLag = Math.min(2000, Math.floor(lineWindow.length/8));
  for(let lag=1; lag<maxLag; lag++){
    let acc=0;
    for(let i=0;i+lag<lineWindow.length;i++) acc += Math.abs(lineWindow[i] - lineWindow[i+lag]);
    if(acc < bestScore){ bestScore = acc; bestLag = lag; }
  }
  let samplesPerTone = Math.max(4, bestLag || Math.max(1, Math.floor(sampleRate*0.01)));

  // compute tones per line and infer toneMultiplier (RGB if ~3x)
  const tonesPerLine = Math.max(1, Math.floor(samplesPerLine / samplesPerTone));
  let toneMultiplier = 1;
  if(tonesPerLine / desiredCols > 2.0) toneMultiplier = 3;

  // compute output rows (use number of detected syncs as upper bound)
  const outCols = desiredCols; const outRows = Math.min(rows, Math.max(1, Math.floor(strongPeaks.length)));
  const imgData = new Uint8ClampedArray(outCols*outRows*4);

  // Decode each line with alignment
  for(let y=0;y<outRows;y++){
    const syncPos = strongPeaks[y] || (firstSync + y*samplesPerLine);
    const lineStart = syncPos + Math.floor(samplesPerTone*0.5);
    for(let x=0;x<outCols;x++){
      const pixelTonePos = Math.floor(x * (tonesPerLine / outCols));
      const toneIndex = pixelTonePos * toneMultiplier;
      const startSample = lineStart + toneIndex * samplesPerTone;
      let r=0,g=0,b=0;
      if(startSample + samplesPerTone >= chan.length) {
        r=g=b=0;
      } else if(toneMultiplier === 1){
        const frame = chan.subarray(startSample, startSample + samplesPerTone);
        const val = detectFrameToGray(frame, sampleRate);
        r=g=b=val;
      } else {
        const vals = [];
        for(let ch=0; ch<toneMultiplier; ch++){
          const frame = chan.subarray(startSample + ch*samplesPerTone, startSample + (ch+1)*samplesPerTone);
          vals.push(detectFrameToGray(frame, sampleRate));
        }
        r = vals[0]||0; g = vals[1]||0; b = vals[2]||0;
      }
      const idx = (y*outCols + x)*4; imgData[idx]=r; imgData[idx+1]=g; imgData[idx+2]=b; imgData[idx+3]=255;
    }
  }
  return {width:outCols, height:outRows, data:imgData};
}

// Helper: fallback simple decode used when sync detection fails
function simpleDecodeFallback(chan, sampleRate, outCols, outRows, options){
  const tonePerPixelMs = options.estimatedToneMs || 10;
  const samplesPerTone = Math.max(4, Math.floor(sampleRate * (tonePerPixelMs/1000)));
  const imgData = new Uint8ClampedArray(outCols*outRows*4);
  let p=0;
  for(let y=0;y<outRows;y++){
    for(let x=0;x<outCols;x++){
      if(p * samplesPerTone >= chan.length){ var gray=0; }
      else {
        const frame = chan.subarray(p*samplesPerTone, p*samplesPerTone + samplesPerTone);
        const val = detectFrameToGray(frame, sampleRate);
        var gray = val;
      }
      const idx = (y*outCols+x)*4; imgData[idx]=imgData[idx+1]=imgData[idx+2]=gray; imgData[idx+3]=255; p++;
    }
  }
  return {width:outCols, height:outRows, data:imgData};
}

// Helper: detect frequency centroid in a frame and map to 0..255 grayscale
function detectFrameToGray(frame, sampleRate){
  const N = frame.length;
  const win = new Float32Array(N); for(let i=0;i<N;i++) win[i]=0.54-0.46*Math.cos(2*Math.PI*i/(N-1));
  const windowed = new Float32Array(N); for(let i=0;i<N;i++) windowed[i]=frame[i]*win[i];
  let weightedSum=0, weightTotal=0;
  // read freq min/max from UI
  const fmin = parseFloat(document.getElementById('freqMin')?.value) || 1500;
  const fmax = parseFloat(document.getElementById('freqMax')?.value) || 2300;
  for(let f=Math.max(800, fmin-200); f<=Math.min(4000, fmax+200); f+=10){ const mag=goertzel(windowed, sampleRate, f); weightedSum+=mag*f; weightTotal+=mag; }
  const freqEstimate = weightTotal>0 ? (weightedSum/weightTotal) : (fmin+fmax)/2;
  const norm = (freqEstimate - fmin)/(fmax-fmin); const clamped = Math.max(0, Math.min(1, norm));
  return Math.floor(clamped*255);
}

function drawDecodedImage(imgObj){
  const canvas = document.getElementById('decodedCanvas');
  canvas.width = imgObj.width; canvas.height = imgObj.height;
  const ctx = canvas.getContext('2d');
  const id = new ImageData(imgObj.data,imgObj.width,imgObj.height);
  ctx.putImageData(id,0,0);
}

function drawLiveImage(imgObj){
  const canvas = document.getElementById('liveDecodedCanvas');
  canvas.width = imgObj.width; canvas.height = imgObj.height;
  const ctx = canvas.getContext('2d');
  const id = new ImageData(imgObj.data,imgObj.width,imgObj.height);
  ctx.putImageData(id,0,0);
}

function startLiveDecoding(analyser, cols=160, rows=128){
  // Prefer buffer-synchronized decoding if we have a generatedBuffer and playingSource
  if(generatedBuffer && playingSource){
    bufferSynchronizedLiveDecode(generatedBuffer, playingSource, cols, rows);
    return;
  }
  if(liveDecodeAnimation) return;
  const bufferLen = analyser.fftSize;
  const floatBuf = new Float32Array(bufferLen);
  // We'll produce a low-res image updated repeatedly. For speed, we'll decode a small block per frame.
  const outCols = Math.min(256, Math.max(32, cols));
  const outRows = Math.min(256, Math.max(32, rows));
  const imgData = new Uint8ClampedArray(outCols*outRows*4);
  let pIndex = 0;
  function step(){
    analyser.getFloatTimeDomainData(floatBuf);
  // decode this frame into a small number of pixels
    const samplesPerTone = Math.max(4, Math.floor(audioCtx.sampleRate * 0.01)); // assume ~10ms per tone
    const numPixelsThisFrame = Math.max(1, Math.floor((floatBuf.length) / samplesPerTone));
    let idx=0;
  // clear only the pixels we will rewrite this frame to avoid cumulative darkening
  // (we'll reset the whole image on each call for simplicity)
  for(let i=0;i<imgData.length;i++) imgData[i]=0;
  for(let i=0;i<numPixelsThisFrame && pIndex < outCols*outRows;i++){
      const start = i * samplesPerTone;
      const frame = floatBuf.subarray(start, Math.min(start+samplesPerTone, floatBuf.length));
      // tiny hamming
      const N=frame.length; const win=new Float32Array(N);
      for(let w=0;w<N;w++) win[w]=0.54-0.46*Math.cos(2*Math.PI*w/(N-1));
      const windowed = new Float32Array(N); for(let w=0;w<N;w++) windowed[w]=frame[w]*win[w];
      // centroid freq
      let weightedSum=0, weightTotal=0;
      for(let f=1200; f<=2400; f+=20){ const mag=goertzel(windowed, audioCtx.sampleRate, f); weightedSum+=mag*f; weightTotal+=mag; }
      const freqEst = weightTotal>0 ? (weightedSum/weightTotal) : 1500;
      const norm = (freqEst - 1500)/(2300-1500); const clamped = Math.max(0, Math.min(1, norm));
      const gray = Math.floor(clamped*255);
      const px = pIndex % outCols; const py = Math.floor(pIndex / outCols);
      const idxx = (py*outCols+px)*4; imgData[idxx]=imgData[idxx+1]=imgData[idxx+2]=gray; imgData[idxx+3]=255;
      pIndex++;
    }
    drawLiveImage({width:outCols, height:outRows, data:imgData});
    // wrap or continue
    if(pIndex >= outCols*outRows) pIndex = 0;
    liveDecodeAnimation = requestAnimationFrame(step);
  }
  liveDecodeAnimation = requestAnimationFrame(step);
}

function bufferSynchronizedLiveDecode(bufObj, sourceNode, cols=320, rows=256){
  // Use buffer metadata to compute tone timing and map playback time to pixel index. This is accurate and avoids analyser drift.
  if(!bufObj || !bufObj.meta) return;
  const meta = bufObj.meta;
  const outCols = meta.cols || cols; const outRows = meta.rows || rows;
  const pixels = outCols * outRows;
  const toneMs = meta.tonePerPixelMs; // per-tone (accounts for RGB multiplier in meta.totalTones)
  const samplesPerTone = meta.samplePerTone || Math.max(1, Math.floor(bufObj.sampleRate * (toneMs/1000)));
  const totalTones = meta.totalTones || (pixels * (meta.toneMultiplier||1));
  const imgData = new Uint8ClampedArray(outCols*outRows*4);
  // clear
  for(let i=0;i<imgData.length;i++) imgData[i]=0;
  function step(){
    const now = audioCtx.currentTime;
    const elapsed = now - playingStartTime; // seconds
    const elapsedMs = elapsed * 1000;
    // compute how many tones have been played so far
    const tonesPlayed = Math.floor(elapsedMs / toneMs);
  // map tones to pixels (toneMultiplier per pixel)
  const toneMultiplier = meta.toneMultiplier || 1;
  const headerOffset = meta.headerOffset || 0;
  const pixelsToShow = Math.min(pixels, Math.floor(tonesPlayed / toneMultiplier));
    // iterate and decode each pixel up to pixelsToShow by reading the buffer directly
    let p = 0;
    for(let y=0;y<outRows;y++){
      for(let x=0;x<outCols;x++){
          if(p >= pixelsToShow){
            // leave black for now
          } else {
          // compute tone index start in samples
          const toneIndex = p * toneMultiplier;
          // read the segment corresponding to first tone of this pixel
          const startSample = headerOffset + toneIndex * samplesPerTone;
          const frame = bufObj.buffer.subarray(startSample, Math.min(startSample+samplesPerTone, bufObj.buffer.length));
          // window
          const N=frame.length; const win=new Float32Array(N);
          for(let w=0;w<N;w++) win[w]=0.54-0.46*Math.cos(2*Math.PI*w/(N-1));
          const windowed = new Float32Array(N); for(let w=0;w<N;w++) windowed[w]=frame[w]*win[w];
          // centroid
          let weightedSum=0, weightTotal=0;
          const fmin = parseFloat(document.getElementById('freqMin')?.value) || 1500;
          const fmax = parseFloat(document.getElementById('freqMax')?.value) || 2300;
          for(let f=Math.max(800,fmin-200); f<=Math.min(4000,fmax+200); f+=10){ const mag=goertzel(windowed, bufObj.sampleRate, f); weightedSum+=mag*f; weightTotal+=mag; }
          const freqEst = weightTotal>0 ? (weightedSum/weightTotal) : (fmin+fmax)/2;
          const norm = (freqEst - fmin)/(fmax-fmin); const clamped = Math.max(0, Math.min(1, norm));
          const gray = Math.floor(clamped*255);
          const idx = (y*outCols+x)*4;
          if(meta.toneMultiplier && meta.toneMultiplier>1){
            // assume sequence R,G,B tones; attempt to decode three tones and assign to channels
            const channelVals = [];
            for(let ch=0; ch<meta.toneMultiplier; ch++){
              const toneIdx = (toneIndex + ch) * samplesPerTone;
              const frameCh = bufObj.buffer.subarray(toneIdx, Math.min(toneIdx+samplesPerTone, bufObj.buffer.length));
              const Nch = frameCh.length; const winch = new Float32Array(Nch);
              for(let w=0;w<Nch;w++) winch[w]=0.54-0.46*Math.cos(2*Math.PI*w/(Nch-1));
              const windowedCh = new Float32Array(Nch); for(let w=0;w<Nch;w++) windowedCh[w]=frameCh[w]*winch[w];
              let weightedSumCh=0, weightTotalCh=0;
              const fmin = parseFloat(document.getElementById('freqMin')?.value) || 1500;
              const fmax = parseFloat(document.getElementById('freqMax')?.value) || 2300;
              for(let f=Math.max(800,fmin-200); f<=Math.min(4000,fmax+200); f+=10){ const mag=goertzel(windowedCh, bufObj.sampleRate, f); weightedSumCh+=mag*f; weightTotalCh+=mag; }
              const freqEstCh = weightTotalCh>0 ? (weightedSumCh/weightTotalCh) : 1500;
              const normCh = (freqEstCh - fmin)/(fmax-fmin); const clampedCh = Math.max(0, Math.min(1, normCh));
              channelVals.push(Math.floor(clampedCh*255));
            }
            imgData[idx]=channelVals[0]||0; imgData[idx+1]=channelVals[1]||0; imgData[idx+2]=channelVals[2]||0; imgData[idx+3]=255;
          } else {
            imgData[idx]=imgData[idx+1]=imgData[idx+2]=gray; imgData[idx+3]=255;
          }
        }
        p++;
      }
    }
    drawLiveImage({width:outCols, height:outRows, data:imgData});
    if(playingSource && playingStartTime && audioCtx.currentTime < playingStartTime + (meta.totalTones * (toneMs/1000))){
      liveDecodeAnimation = requestAnimationFrame(step);
    } else {
      // playback finished, stop
      stopLiveDecoding();
    }
  }
  stopLiveDecoding();
  liveDecodeAnimation = requestAnimationFrame(step);
}

function stopLiveDecoding(){
  if(liveDecodeAnimation) cancelAnimationFrame(liveDecodeAnimation);
  liveDecodeAnimation = null;
  if(liveDecodeAnalyser){ try{ masterGain.disconnect(liveDecodeAnalyser); }catch(e){} liveDecodeAnalyser=null; }
}

// UI wiring
document.getElementById('imgFile').addEventListener('change', async (e)=>{
  const f = e.target.files[0]; if(!f) return;
  const img = await fileToImage(f);
  drawPreview(img);
  window._lastImage = img;
});

document.getElementById('encodeBtn').addEventListener('click', async ()=>{
  const img = window._lastImage; if(!img) { alert('Select an image first'); return; }
  const sr = parseInt(document.getElementById('sampleRate').value) || 44100;
  const duration = parseInt(document.getElementById('encodeDuration').value) || 10;
  // Determine appropriate resolution so the total pixels fit the duration given tonePerPixelMs
  // We'll assume tonePerPixelMs will be computed inside synthesize; approximate using 10ms base.
  const approxToneMs = Math.max(1, Math.min(200, (duration*1000*0.98) / (320*256)));
  // Compute desired total pixels = durationSeconds * 1000 / toneMs
  const totalMs = duration*1000*0.98;
  const desiredToneMs = Math.max(1, Math.min(200, totalMs / (32*32))); // ensure minimum image of 32x32
  const desiredTotalPixels = Math.floor(totalMs / desiredToneMs);
  // choose cols such that cols*rows ~= desiredTotalPixels, keep aspect ratio of image
  const aspect = img.width / img.height;
  // choose rows around sqrt(totalPixels/aspect)
  let rows = Math.max(32, Math.round(Math.sqrt(desiredTotalPixels / aspect)));
  let cols = Math.max(32, Math.round(rows * aspect));
  // clamp to reasonable maxima
  rows = Math.min(256, rows); cols = Math.min(512, cols);
  const samples = imageToGrayscaleSamples(img, cols, rows);
  const bufObj = synthesizeSSTV(samples, sr, {durationSeconds: duration});
  generatedBuffer = bufObj;
  playingOffsetSeconds = 0; playingPaused = false;
  const src = playFloat32Buffer(bufObj);
  // If live decode is enabled, wire an analyser to the playing node
  const liveToggle = document.getElementById('liveDecodeToggle');
  if(liveToggle && liveToggle.checked){
    // create a node from generated buffer and connect to master gain; use an analyser to read from masterGain
    try{
      if(liveDecodeAnalyser) { /* reuse */ }
      else {
        liveDecodeAnalyser = audioCtx.createAnalyser(); liveDecodeAnalyser.fftSize = 2048;
        ensureAudioCtx(); masterGain.connect(liveDecodeAnalyser);
      }
      startLiveDecoding(liveDecodeAnalyser, cols || 320, rows || 256);
    }catch(e){ console.warn('live decode setup failed', e); }
  }
  src.onended = ()=>{ console.log('playback ended'); };
  // enable pause/stop
  document.getElementById('pauseBtn').disabled = false;
  document.getElementById('stopBtn').disabled = false;
  document.getElementById('downloadWavBtn').disabled=false;
});

document.getElementById('pauseBtn').addEventListener('click', ()=>{
  if(!playingSource) return;
  if(!playingPaused){
    // pause: stop source and record offset
    const elapsed = audioCtx.currentTime - playingStartTime;
    try{ playingSource.stop(); }catch(e){}
    playingOffsetSeconds = elapsed;
    playingPaused = true;
    document.getElementById('pauseBtn').innerText = 'Resume';
  } else {
    // resume
    const src = playFloat32Buffer(generatedBuffer);
    playingSource = src;
    playingPaused = false;
    document.getElementById('pauseBtn').innerText = 'Pause';
  }
});

document.getElementById('stopBtn').addEventListener('click', ()=>{
  if(playingSource){ try{ playingSource.stop(); }catch(e){} }
  playingSource = null; playingOffsetSeconds = 0; playingPaused = false;
  document.getElementById('pauseBtn').disabled = true; document.getElementById('stopBtn').disabled = true; document.getElementById('pauseBtn').innerText='Pause';
  stopLiveDecoding();
});

document.getElementById('downloadWavBtn').addEventListener('click', ()=>{
  if(!generatedBuffer) return; const blob = floatToWav(generatedBuffer.buffer, generatedBuffer.sampleRate);
  const a = document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='sstv_emulator.wav'; a.click();
});

// decode uploaded audio
document.getElementById('audioFile').addEventListener('change', (e)=>{ window._lastAudioFile = e.target.files[0]; });

document.getElementById('decodeBtn').addEventListener('click', async ()=>{
  const f = window._lastAudioFile; if(!f) { alert('Select an audio file first'); return; }
  const duration = parseInt(document.getElementById('decodeDuration').value) || 10;
  const out = await decodeAudioBlob(f,320,256, {durationSeconds: duration, estimatedToneMs:  Math.max(1, (duration*1000*0.98) / (320*256))});
  drawDecodedImage(out);
});

// Microphone listen
let micStream=null, micSource=null, analyser=null, micProcessor=null;

document.getElementById('micListenBtn').addEventListener('click', async ()=>{
  ensureAudioCtx();
  if(micStream) return;
  try{
    micStream = await navigator.mediaDevices.getUserMedia({audio:true});
  }catch(err){ alert('Microphone access denied or unavailable'); return; }
  micSource = audioCtx.createMediaStreamSource(micStream);
  analyser = audioCtx.createAnalyser(); analyser.fftSize=2048;
  micSource.connect(analyser);
  const data = new Float32Array(analyser.fftSize);
  document.getElementById('stopMicBtn').disabled=false;
  document.getElementById('micListenBtn').disabled=true;
  // simple live visualization
  function draw(){
    analyser.getFloatTimeDomainData(data);
    const viz = document.getElementById('viz'); const vctx = viz.getContext('2d');
    vctx.fillStyle='#fff'; vctx.fillRect(0,0,viz.width,viz.height);
    vctx.strokeStyle='#007'; vctx.beginPath();
    for(let i=0;i<data.length;i++){
      const x = (i/data.length)*viz.width; const y = (0.5+data[i]*0.5)*viz.height;
      if(i===0) vctx.moveTo(x,y); else vctx.lineTo(x,y);
    }
    vctx.stroke();
    micProcessor = requestAnimationFrame(draw);
  }
  draw();
});

document.getElementById('stopMicBtn').addEventListener('click', ()=>{
  if(micStream){
    micStream.getTracks().forEach(t=>t.stop()); micStream=null;
  }
  if(micProcessor) cancelAnimationFrame(micProcessor);
  micProcessor=null; document.getElementById('stopMicBtn').disabled=true; document.getElementById('micListenBtn').disabled=false;
});

// Volume control
const volumeControl = document.getElementById('volumeControl');
if(volumeControl){
  volumeControl.addEventListener('input', (e)=>{
    ensureAudioCtx();
    const v = parseFloat(e.target.value);
    if(masterGain) masterGain.gain.setValueAtTime(v, audioCtx.currentTime);
  });
}

// Live decode UI wiring
const liveToggle = document.getElementById('liveDecodeToggle');
const startLiveBtn = document.getElementById('startLiveDecodeBtn');
const stopLiveBtn = document.getElementById('stopLiveDecodeBtn');
if(liveToggle){
  liveToggle.addEventListener('change', ()=>{
    if(liveToggle.checked){ startLiveBtn.disabled=false; } else { startLiveBtn.disabled=true; stopLiveBtn.disabled=true; stopLiveDecoding(); }
  });
}
if(startLiveBtn){ startLiveBtn.addEventListener('click', ()=>{ if(!liveDecodeAnalyser){ liveDecodeAnalyser = audioCtx.createAnalyser(); liveDecodeAnalyser.fftSize=2048; masterGain.connect(liveDecodeAnalyser); } startLiveDecoding(liveDecodeAnalyser,320,256); startLiveBtn.disabled=true; stopLiveBtn.disabled=false; }); }
if(stopLiveBtn){ stopLiveBtn.addEventListener('click', ()=>{ stopLiveDecoding(); startLiveBtn.disabled=false; stopLiveBtn.disabled=true; }); }

// small note: this is an educational emulator. Real SSTV uses precise timing, sync pulses, multi-tone encoding per line and color channels.
