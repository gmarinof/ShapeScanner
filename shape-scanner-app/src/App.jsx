import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Upload, Check, RefreshCcw, Settings, Download, ScanLine, ZoomIn, ZoomOut, Maximize2, MousePointer2, Eye, EyeOff, Sun, Palette, Pipette, ToggleLeft, ToggleRight, AlertTriangle, Image as ImageIcon, Layers, Flame, Bug, PenTool, FileText, CreditCard, BoxSelect, Eraser, RotateCcw, Sparkles, X } from 'lucide-react';

/**
 * MATH HELPERS
 */
class Homography {
  static getTransform(src, dst) {
    if (src.length !== 4 || dst.length !== 4) return [1,0,0,0,1,0,0,0,1];
    let a = [];
    for (let i = 0; i < 4; i++) {
      a.push([src[i].x, src[i].y, 1, 0, 0, 0, -src[i].x * dst[i].x, -src[i].y * dst[i].x]);
      a.push([0, 0, 0, src[i].x, src[i].y, 1, -src[i].x * dst[i].y, -src[i].y * dst[i].y]);
    }
    const b = [];
    for (let i = 0; i < 4; i++) { b.push(dst[i].x); b.push(dst[i].y); }
    try {
        const h = this.solveGaussian(a, b);
        return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
    } catch(e) { return [1,0,0,0,1,0,0,0,1]; }
  }

  static solveGaussian(A, b) {
    const n = A.length;
    for (let i = 0; i < n; i++) {
      let maxEl = Math.abs(A[i][i]);
      let maxRow = i;
      for (let k = i + 1; k < n; k++) { if (Math.abs(A[k][i]) > maxEl) { maxEl = Math.abs(A[k][i]); maxRow = k; } }
      const tmpRow = A[maxRow]; A[maxRow] = A[i]; A[i] = tmpRow;
      const tmpB = b[maxRow]; b[maxRow] = b[i]; b[i] = tmpB;
      for (let k = i + 1; k < n; k++) {
        const c = -A[k][i] / A[i][i];
        for (let j = i; j < n; j++) { if (i === j) { A[k][j] = 0; } else { A[k][j] += c * A[i][j]; } }
        b[k] += c * b[i];
      }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i > -1; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) { sum += A[i][j] * x[j]; }
      x[i] = (b[i] - sum) / A[i][i];
    }
    return x;
  }

  static transformPoint(x, y, H) {
    const newX = (H[0] * x + H[1] * y + H[2]) / (H[6] * x + H[7] * y + H[8]);
    const newY = (H[3] * x + H[4] * y + H[5]) / (H[6] * x + H[7] * y + H[8]);
    return { x: newX, y: newY };
  }
}

/**
 * VECTOR HELPERS
 */
class VectorUtils {
    static getSqSegDist(p, p1, p2) {
        let x = p1.x, y = p1.y, dx = p2.x - x, dy = p2.y - y;
        if (dx !== 0 || dy !== 0) {
            const t = ((p.x - x) * dx + (p.y - y) * dy) / (dx * dx + dy * dy);
            if (t > 1) { x = p2.x; y = p2.y; }
            else if (t > 0) { x += dx * t; y += dy * t; }
        }
        dx = p.x - x; dy = p.y - y;
        return dx * dx + dy * dy;
    }

    static simplify(points, sqTolerance) {
        const len = points.length;
        if (len <= 2) return points;
        let maxSqDist = 0;
        let index = 0;
        for (let i = 1; i < len - 1; i++) {
            const sqDist = this.getSqSegDist(points[i], points[0], points[len - 1]);
            if (sqDist > maxSqDist) { maxSqDist = sqDist; index = i; }
        }
        if (maxSqDist > sqTolerance) {
            const left = this.simplify(points.slice(0, index + 1), sqTolerance);
            const right = this.simplify(points.slice(index), sqTolerance);
            return left.slice(0, left.length - 1).concat(right);
        }
        return [points[0], points[len - 1]];
    }

    static smooth(points, iterations = 1) {
        if (points.length < 3 || iterations < 1) return points;
        let output = [...points];
        for (let i = 0; i < iterations; i++) {
            const newPoints = [];
            output.push(output[0]); 
            for (let j = 0; j < output.length - 1; j++) {
                const p0 = output[j];
                const p1 = output[j + 1];
                const Q = { x: 0.75 * p0.x + 0.25 * p1.x, y: 0.75 * p0.y + 0.25 * p1.y };
                const R = { x: 0.25 * p0.x + 0.75 * p1.x, y: 0.25 * p0.y + 0.75 * p1.y };
                newPoints.push(Q, R);
            }
            output = newPoints;
        }
        return output;
    }
}

/**
 * SHAPE FITTER
 */
class ShapeFitter {
    static fitCircle(points) {
        if (points.length < 10) return null; 
        let sumX = 0, sumY = 0;
        points.forEach(p => { sumX += p.x; sumY += p.y; });
        const cx = sumX / points.length;
        const cy = sumY / points.length;
        let sumR = 0;
        points.forEach(p => { sumR += Math.sqrt(Math.pow(p.x - cx, 2) + Math.pow(p.y - cy, 2)); });
        const r = sumR / points.length;
        let sqErr = 0;
        points.forEach(p => {
            const d = Math.sqrt(Math.pow(p.x - cx, 2) + Math.pow(p.y - cy, 2));
            sqErr += Math.pow(d - r, 2);
        });
        const rmse = Math.sqrt(sqErr / points.length);
        const relativeError = rmse / r; 
        if (relativeError < 0.06) return { type: 'circle', cx, cy, r };
        return null;
    }

    static generateCircle(cx, cy, r, steps=64) {
        const pts = [];
        for(let i=0; i<steps; i++) {
            const theta = (i / steps) * Math.PI * 2;
            pts.push({ x: cx + r * Math.cos(theta), y: cy + r * Math.sin(theta) });
        }
        pts.push(pts[0]); 
        return pts;
    }
}

/**
 * AI SERVICE
 */
const callGeminiVision = async (base64Image, width, height) => {
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY || "";
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
    
    if (!apiKey) {
        return "AI analysis unavailable: No API key configured. Set VITE_GEMINI_API_KEY in your environment.";
    }

    const prompt = `Analyze this technical shape contour (black shape on white background). The image dimensions are ${width}mm x ${height}mm.
    1. Identify what this part likely is (e.g. gasket, bracket, shim, etc).
    2. Suggest the best manufacturing method (Laser cutting, CNC, 3D printing).
    3. Suggest common materials for this shape.
    Keep response concise and practical for a machinist/engineer.`;

    const payload = {
        contents: [{
            parts: [
                { text: prompt },
                { inlineData: { mimeType: "image/png", data: base64Image } }
            ]
        }]
    };

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error(`API Error: ${response.status}`);

        const data = await response.json();
        return data.candidates?.[0]?.content?.parts?.[0]?.text || "No analysis generated.";
    } catch (error) {
        console.error("Gemini API Error:", error);
        return "Error analyzing shape. Please check connection.";
    }
};

/**
 * ANIMATED SPLASH SCREEN COMPONENT
 */
const SplashScreen = ({ onFinish }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onFinish();
        }, 2800);
        return () => clearTimeout(timer);
    }, [onFinish]);

    return (
        <div className="fixed inset-0 bg-black flex flex-col items-center justify-center z-[100]">
             <style>{`
                @keyframes scan-beam {
                    0%, 100% { top: 0%; opacity: 0; }
                    10% { opacity: 1; }
                    50% { top: 100%; opacity: 1; }
                    90% { opacity: 1; }
                }
            `}</style>
            <div className="relative w-24 h-24 mb-6">
                <div className="absolute inset-0 border-4 border-neutral-700 rounded-lg"></div>
                <div 
                    className="absolute top-0 left-0 w-full h-1 bg-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.8)]"
                    style={{ animation: 'scan-beam 2s ease-in-out infinite' }}
                ></div>
                <div className="absolute -top-1 -left-1 w-6 h-6 border-t-4 border-l-4 border-white rounded-tl-lg"></div>
                <div className="absolute -top-1 -right-1 w-6 h-6 border-t-4 border-r-4 border-white rounded-tr-lg"></div>
                <div className="absolute -bottom-1 -left-1 w-6 h-6 border-b-4 border-l-4 border-white rounded-bl-lg"></div>
                <div className="absolute -bottom-1 -right-1 w-6 h-6 border-b-4 border-r-4 border-white rounded-br-lg"></div>
            </div>
            
            <h1 className="text-3xl font-bold text-white tracking-widest animate-[pulse_3s_ease-in-out]">
                SHAPE<span className="text-blue-500">SCANNER</span>
            </h1>
            <p className="text-neutral-500 text-xs mt-2 tracking-[0.3em] uppercase">Vectorization Engine</p>
        </div>
    );
};

/**
 * MAIN COMPONENT
 */
const ShapeScanner = () => {
  // --- SPLASH STATE ---
  const [showSplash, setShowSplash] = useState(true);

  // --- APP STATE ---
  const [step, setStep] = useState('capture');
  const [imageSrc, setImageSrc] = useState(null);
  const [imgDims, setImgDims] = useState({ w: 0, h: 0 });
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Debug State
  const [showDebug, setShowDebug] = useState(false);
  const [debugStats, setDebugStats] = useState({ mappedPixels: 0, matrixValid: false, processingTime: 0 });

  // Calibration
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [calMonochrome, setCalMonochrome] = useState(false);
  const [calContrast, setCalContrast] = useState(100);
  const [corners, setCorners] = useState([]); 
  const [activeCorner, setActiveCorner] = useState(null);
  const [paperWidth, setPaperWidth] = useState(210); 
  const [paperHeight, setPaperHeight] = useState(297); 
  const [orientation, setOrientation] = useState('portrait'); 
  
  // Processing
  const [threshold, setThreshold] = useState(30); 
  const [scanStep, setScanStep] = useState(2); 
  const [curveSmoothing, setCurveSmoothing] = useState(2); 
  const [noiseFilter, setNoiseFilter] = useState(2); 
  const [smartRefine, setSmartRefine] = useState(true); 
  const [shadowRemoval, setShadowRemoval] = useState(0);
  const [processedPath, setProcessedPath] = useState([]); 
  const [objectDims, setObjectDims] = useState(null);
  const [showMask, setShowMask] = useState(false); 
  const [invertResult, setInvertResult] = useState(false); 
  const [viewMode, setViewMode] = useState('original'); 
  const [detectedShapeType, setDetectedShapeType] = useState(null);
  
  // AI
  const [aiResult, setAiResult] = useState(null);
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [showAiPanel, setShowAiPanel] = useState(false);
  
  const [segmentMode, setSegmentMode] = useState('auto'); 
  const [targetColor, setTargetColor] = useState({r:255, g:255, b:255}); 
  const [calculatedRefColor, setCalculatedRefColor] = useState({r:0, g:0, b:0}); 
  
  // Interaction
  const [selectionBox, setSelectionBox] = useState(null); 
  const [dragStart, setDragStart] = useState(null);
  const [isPicking, setIsPicking] = useState(false);

  // Refs
  const canvasRef = useRef(null);
  const processCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const sourcePixelData = useRef(null); 
  const unwarpedBufferRef = useRef(null);

  // --- HELPER: Coordinate Mapping ---
  const toImageCoords = (screenX, screenY) => ({
    x: (screenX - view.x) / view.scale,
    y: (screenY - view.y) / view.scale
  });

  // --- STEP 1: CAPTURE & LOAD ---
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        const img = new Image();
        img.onload = () => {
          const maxDim = 1500; 
          let w = img.width; let h = img.height;
          if (w > maxDim || h > maxDim) {
            const scale = Math.min(maxDim / w, maxDim / h);
            w = Math.floor(w * scale); h = Math.floor(h * scale);
          }
          
          setCorners([
            { x: w * 0.2, y: h * 0.2 }, { x: w * 0.8, y: h * 0.2 },
            { x: w * 0.8, y: h * 0.8 }, { x: w * 0.2, y: h * 0.8 },
          ]);
          setImgDims({ w, h });
          setImageSrc(evt.target.result);
          
          // Reset
          setCalMonochrome(false); setCalContrast(100);
          setSegmentMode('auto'); setInvertResult(false);
          setProcessedPath([]); setObjectDims(null);
          setViewMode('original'); 
          setAiResult(null); setShowAiPanel(false);
          
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = w; tempCanvas.height = h;
          const ctx = tempCanvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);
          sourcePixelData.current = ctx.getImageData(0, 0, w, h).data;
          
          setStep('calibrate');
        };
        img.src = evt.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  // --- HELPER: PRESET MANAGER ---
  const applyPreset = (w, h) => {
      if (orientation === 'portrait') {
          setPaperWidth(Math.min(w, h));
          setPaperHeight(Math.max(w, h));
      } else {
          setPaperWidth(Math.max(w, h));
          setPaperHeight(Math.min(w, h));
      }
  };

  const toggleOrientation = () => {
      const newOri = orientation === 'portrait' ? 'landscape' : 'portrait';
      setOrientation(newOri);
      setPaperWidth(paperHeight);
      setPaperHeight(paperWidth);
  };

  // --- STEP 2: CALIBRATE INTERACTION ---
  const fitToScreen = useCallback(() => {
    if (!canvasRef.current || imgDims.w === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scale = Math.min(rect.width / imgDims.w, rect.height / imgDims.h) * 0.9; 
    const x = (rect.width - imgDims.w * scale) / 2;
    const y = (rect.height - imgDims.h * scale) / 2;
    setView({ x, y, scale });
  }, [imgDims]);

  useEffect(() => { if (step === 'calibrate') setTimeout(fitToScreen, 100); }, [step, fitToScreen]);

  const handleStart = (clientX, clientY) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = clientX - rect.left; const sy = clientY - rect.top;
    const p = toImageCoords(sx, sy);
    const hitRadius = 30 / view.scale; 
    let closest = -1; let minD = Infinity;
    corners.forEach((c, i) => {
      const d = Math.sqrt((c.x - p.x)**2 + (c.y - p.y)**2);
      if (d < hitRadius && d < minD) { minD = d; closest = i; }
    });
    if (closest !== -1) setActiveCorner(closest);
    else { setIsPanning(true); setLastMousePos({ x: clientX, y: clientY }); }
  };

  const handleMove = (clientX, clientY) => {
    if (activeCorner !== null) {
      const rect = canvasRef.current.getBoundingClientRect();
      const sx = clientX - rect.left; const sy = clientY - rect.top;
      const p = toImageCoords(sx, sy);
      setCorners(prev => {
        const newCorners = [...prev];
        newCorners[activeCorner] = { x: Math.max(0, Math.min(imgDims.w, p.x)), y: Math.max(0, Math.min(imgDims.h, p.y)) };
        return newCorners;
      });
    } else if (isPanning) {
      const dx = clientX - lastMousePos.x; const dy = clientY - lastMousePos.y;
      setView(v => ({ ...v, x: v.x + dx, y: v.y + dy }));
      setLastMousePos({ x: clientX, y: clientY });
    }
  };

  const handleEnd = () => { setActiveCorner(null); setIsPanning(false); };
  
  const handleWheel = (e) => {
    if (step !== 'calibrate') return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left; const my = e.clientY - rect.top;
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.1, Math.min(10, view.scale * zoomFactor));
    const newX = mx - (mx - view.x) * (newScale / view.scale);
    const newY = my - (my - view.y) * (newScale / view.scale);
    setView({ x: newX, y: newY, scale: newScale });
  };

  useEffect(() => {
    if (step === 'calibrate' && canvasRef.current && imageSrc) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = new Image(); img.src = imageSrc;
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== rect.width || canvas.height !== rect.height) { canvas.width = rect.width; canvas.height = rect.height; }
      
      img.onload = () => {
        ctx.fillStyle = '#000000'; // Pure black
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(view.x, view.y);
        ctx.scale(view.scale, view.scale);
        ctx.filter = `grayscale(${calMonochrome ? 100 : 0}%) contrast(${calContrast}%)`;
        ctx.drawImage(img, 0, 0, imgDims.w, imgDims.h);
        ctx.filter = 'none';
        ctx.fillStyle = 'rgba(0,0,0,0.6)'; // Darker overlay for better contrast
        ctx.fillRect(0, 0, imgDims.w, imgDims.h);
        ctx.beginPath();
        ctx.moveTo(corners[0].x, corners[0].y); ctx.lineTo(corners[1].x, corners[1].y);
        ctx.lineTo(corners[2].x, corners[2].y); ctx.lineTo(corners[3].x, corners[3].y);
        ctx.closePath();
        ctx.lineWidth = 2 / view.scale; ctx.strokeStyle = '#3b82f6'; ctx.stroke(); // Blue line
        ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'; ctx.fill();
        
        const labels = ['TL', 'TR', 'BR', 'BL'];
        const handleRadius = 12 / view.scale;
        corners.forEach((c, i) => {
            ctx.beginPath(); ctx.arc(c.x, c.y, handleRadius, 0, 2 * Math.PI);
            ctx.fillStyle = '#3b82f6'; ctx.fill();
            ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2/view.scale; ctx.stroke();
            ctx.fillStyle = 'white'; ctx.font = `bold ${12/view.scale}px sans-serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(labels[i], c.x, c.y);
        });
        ctx.restore();
      };
    }
  }, [corners, step, imageSrc, imgDims, view, calMonochrome, calContrast]);

  // --- OTSU AUTO DETECT ---
  const autoDetectCorners = () => {
    if (!sourcePixelData.current) return;
    const width = imgDims.w; const height = imgDims.h; const data = sourcePixelData.current;
    
    const histogram = new Array(256).fill(0);
    const step = 4;
    for(let i=0; i<data.length; i+=4*step) {
        const lum = Math.round(0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2]);
        histogram[lum]++;
    }

    let sum = 0; for (let i = 0; i < 256; i++) sum += i * histogram[i];
    let sumB = 0; let wB = 0; let wF = 0; let maxVar = 0; let otsuThreshold = 0;
    const total = (data.length / 4) / step;

    for (let i = 0; i < 256; i++) {
        wB += histogram[i]; if (wB === 0) continue;
        wF = total - wB; if (wF === 0) break;
        sumB += i * histogram[i];
        const mB = sumB / wB; const mF = (sum - sumB) / wF;
        const varBetween = wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) { maxVar = varBetween; otsuThreshold = i; }
    }

    let tl = { val: Infinity, x: 0, y: 0 }; let tr = { val: -Infinity, x: 0, y: 0 };
    let br = { val: -Infinity, x: 0, y: 0 }; let bl = { val: Infinity, x: 0, y: 0 };
    const padding = Math.min(width, height) * 0.05; 

    for (let y = padding; y < height - padding; y += step) {
      for (let x = padding; x < width - padding; x += step) {
        const i = (Math.floor(y) * width + Math.floor(x)) * 4;
        const brightness = 0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2];
        
        if (brightness > otsuThreshold) { 
          const sum = x + y; const diff = x - y;
          if (sum < tl.val) { tl.val = sum; tl.x = x; tl.y = y; }
          if (diff > tr.val) { tr.val = diff; tr.x = x; tr.y = y; }
          if (sum > br.val) { br.val = sum; br.x = x; br.y = y; }
          if (diff < bl.val) { bl.val = diff; bl.x = x; bl.y = y; }
        }
      }
    }
    if (tl.val !== Infinity) { setCorners([{ x: tl.x, y: tl.y }, { x: tr.x, y: tr.y }, { x: br.x, y: br.y }, { x: bl.x, y: bl.y }]); }
  };


  // --- PROCESSING STEP INTERACTION ---
  const handleProcessStart = (e) => {
    const rect = processCanvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left; const y = e.clientY - rect.top;
    const scaleX = processCanvasRef.current.width / rect.width;
    const scaleY = processCanvasRef.current.height / rect.height;
    const px = x * scaleX; const py = y * scaleY;

    if (isPicking) {
        if (unwarpedBufferRef.current) {
            const { width, data } = unwarpedBufferRef.current;
            const ix = Math.floor(px); const iy = Math.floor(py);
            if(ix >= 0 && ix < width && iy >= 0) {
                 const i = (iy * width + ix) * 4;
                 if (data[i+3] > 0) {
                    setTargetColor({ r: data[i], g: data[i+1], b: data[i+2] });
                    setSegmentMode('manual-bg'); setIsPicking(false); setViewMode('heatmap'); 
                 }
            }
        }
        return;
    }
    setDragStart({ x: px, y: py });
    setSelectionBox(null);
  };

  const handleProcessMove = (e) => {
    if (!dragStart || isPicking) return;
    const rect = processCanvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (processCanvasRef.current.width / rect.width);
    const y = (e.clientY - rect.top) * (processCanvasRef.current.height / rect.height);
    setSelectionBox({ x: dragStart.x, y: dragStart.y, w: x - dragStart.x, h: y - dragStart.y });
  };

  const handleProcessEnd = () => {
    if (isPicking) return;
    if (!dragStart || !selectionBox) { setDragStart(null); return; }

    if (unwarpedBufferRef.current) {
        const { width, height, data } = unwarpedBufferRef.current;
        let r = 0, g = 0, b = 0, count = 0;
        const startX = Math.floor(Math.min(dragStart.x, dragStart.x + selectionBox.w));
        const startY = Math.floor(Math.min(dragStart.y, dragStart.y + selectionBox.h));
        const endX = Math.ceil(Math.max(dragStart.x, dragStart.x + selectionBox.w));
        const endY = Math.ceil(Math.max(dragStart.y, dragStart.y + selectionBox.h));
        for(let y = startY; y < endY; y++) {
            for(let x = startX; x < endX; x++) {
                if(x >= 0 && x < width && y >= 0 && y < height) {
                    const i = (y * width + x) * 4;
                    if (data[i+3] > 0) { r += data[i]; g += data[i+1]; b += data[i+2]; count++; }
                }
            }
        }
        if (count > 0) {
            setTargetColor({ r: r/count, g: g/count, b: b/count });
            setSegmentMode('manual-obj'); setViewMode('heatmap'); 
        }
    }
    setDragStart(null); setSelectionBox(null);
  };


  // --- MAIN PROCESS LOGIC ---
  const processImage = useCallback(() => {
    if (!processCanvasRef.current || !sourcePixelData.current) return;
    const startTime = performance.now();
    setIsProcessing(true);

    const canvas = processCanvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    const pixelsPerMM = 2; 
    const targetW = Math.floor(paperWidth * pixelsPerMM);
    const targetH = Math.floor(paperHeight * pixelsPerMM);
    
    if(canvas.width !== targetW || canvas.height !== targetH) { canvas.width = targetW; canvas.height = targetH; }

    const dstCoords = [{ x: 0, y: 0 }, { x: targetW, y: 0 }, { x: targetW, y: targetH }, { x: 0, y: targetH }];
    const matrix = Homography.getTransform(dstCoords, corners);
    const imgData = ctx.createImageData(targetW, targetH);
    const data = imgData.data;
    const srcData = sourcePixelData.current; 
    const rawBuffer = new Uint8ClampedArray(targetW * targetH * 4);

    let refR = 255, refG = 255, refB = 255;
    
    if (segmentMode === 'auto') {
        const cx = (corners[0].x + corners[1].x + corners[2].x + corners[3].x)/4;
        const cy = (corners[0].y + corners[1].y + corners[2].y + corners[3].y)/4;
        let bgR = 0, bgG = 0, bgB = 0, count = 0;
        corners.forEach(c => {
            const sampleX = c.x + (cx - c.x) * 0.15; 
            const sampleY = c.y + (cy - c.y) * 0.15;
            for(let dy = -1; dy <= 1; dy++) {
                for(let dx = -1; dx <= 1; dx++) {
                    const sx = Math.max(0, Math.min(imgDims.w - 1, Math.round(sampleX) + dx));
                    const sy = Math.max(0, Math.min(imgDims.h - 1, Math.round(sampleY) + dy));
                    const i = (sy * imgDims.w + sx) * 4;
                    bgR += srcData[i]; bgG += srcData[i+1]; bgB += srcData[i+2]; count++;
                }
            }
        });
        if(count>0) { refR=bgR/count; refG=bgG/count; refB=bgB/count; }
    } else {
        refR = targetColor.r; refG = targetColor.g; refB = targetColor.b;
    }

    const points = [];
    let mappedPixelCount = 0;
    
    // BINARY MASK BUFFER
    const mask = new Uint8Array(targetW * targetH); 

    for (let y = 0; y < targetH; y++) {
      for (let x = 0; x < targetW; x++) {
        const idx = (y * targetW + x) * 4;
        const srcPt = Homography.transformPoint(x, y, matrix);
        const srcX = Math.round(srcPt.x);
        const srcY = Math.round(srcPt.y);

        if (srcX >= 0 && srcX < imgDims.w && srcY >= 0 && srcY < imgDims.h) {
          mappedPixelCount++;
          const srcIdx = (srcY * imgDims.w + srcX) * 4;
          const r = srcData[srcIdx];
          const g = srcData[srcIdx + 1];
          const b = srcData[srcIdx + 2];
          
          rawBuffer[idx] = r; rawBuffer[idx+1] = g; rawBuffer[idx+2] = b; rawBuffer[idx+3] = 255;

          // NOISE FILTER
          let testR = r, testG = g, testB = b;
          if (noiseFilter > 0) {
              if (srcX > 0 && srcX < imgDims.w - 1 && srcY > 0 && srcY < imgDims.h - 1) {
                  let nr=r, ng=g, nb=b;
                  const off = 4; const w4 = imgDims.w * 4;
                  nr += srcData[srcIdx-off] + srcData[srcIdx+off] + srcData[srcIdx-w4] + srcData[srcIdx+w4];
                  ng += srcData[srcIdx+1-off] + srcData[srcIdx+1+off] + srcData[srcIdx+1-w4] + srcData[srcIdx+1+w4];
                  nb += srcData[srcIdx+2-off] + srcData[srcIdx+2+off] + srcData[srcIdx+2-w4] + srcData[srcIdx+2+w4];
                  testR = nr / 5; testG = ng / 5; testB = nb / 5;
              }
          }

          const dist = Math.sqrt((refR-testR)**2 + (refG-testG)**2 + (refB-testB)**2);
          let isObject = false;
          
          if (segmentMode === 'auto' || segmentMode === 'manual-bg') { isObject = dist > threshold; }
          else if (segmentMode === 'manual-obj') { isObject = dist < threshold; }
          if (invertResult) isObject = !isObject;

          mask[y * targetW + x] = isObject ? 1 : 0;
        } else {
          rawBuffer[idx]=0; rawBuffer[idx+1]=0; rawBuffer[idx+2]=0; rawBuffer[idx+3]=0;
          mask[y * targetW + x] = 0;
        }
      }
    }

    // --- MORPHOLOGICAL EROSION ---
    if (shadowRemoval > 0) {
        const erosionIterations = Math.floor(shadowRemoval);
        for(let iter=0; iter<erosionIterations; iter++) {
            const erodedMask = new Uint8Array(targetW * targetH);
            for(let y=1; y<targetH-1; y++) {
                for(let x=1; x<targetW-1; x++) {
                    const i = y * targetW + x;
                    if (mask[i] === 1) {
                        if (mask[i-1]===0 || mask[i+1]===0 || mask[i-targetW]===0 || mask[i+targetW]===0) {
                            erodedMask[i] = 0;
                        } else {
                            erodedMask[i] = 1;
                        }
                    }
                }
            }
            mask.set(erodedMask); 
        }
    }

    // --- DRAWING FINAL PIXELS ---
    for (let y = 0; y < targetH; y++) {
        for (let x = 0; x < targetW; x++) {
            const idx = (y * targetW + x) * 4;
            const isObject = mask[y * targetW + x] === 1;
            const r = rawBuffer[idx]; const g = rawBuffer[idx+1]; const b = rawBuffer[idx+2];

            if (viewMode === 'original') {
                data[idx] = r; data[idx+1] = g; data[idx+2] = b; data[idx+3] = 255;
            } else if (viewMode === 'contour') {
                data[idx] = 0; data[idx+1] = 0; data[idx+2] = 0; data[idx+3] = 0;
            } else if (viewMode === 'heatmap') {
                const dist = Math.sqrt((refR-r)**2 + (refG-g)**2 + (refB-b)**2);
                const intensity = Math.min(255, dist * 3); 
                data[idx] = intensity; data[idx+1] = intensity > 128 ? 255 - intensity : 0; data[idx+2] = 0; data[idx+3] = 255; 
            } else {
                // Processed View
                if (showMask) {
                    const val = isObject ? 255 : 0;
                    data[idx] = val; data[idx+1] = val; data[idx+2] = val; data[idx+3] = 255;
                } else {
                    if (isObject) {
                        data[idx] = r; data[idx+1] = g; data[idx+2] = b; data[idx+3] = 255;
                    } else {
                        data[idx] = 0; data[idx+1]=0; data[idx+2]=0; data[idx+3]=0;
                    }
                }
            }
        }
    }
    
    if (mappedPixelCount === 0) {
        ctx.fillStyle = "red"; ctx.font = "20px Arial"; ctx.fillText("MAPPING FAILED", 10, 50);
    } else {
        ctx.putImageData(imgData, 0, 0);
    }
    
    unwarpedBufferRef.current = { width: targetW, height: targetH, data: rawBuffer };

    // Calculate Bounds
    if (viewMode !== 'original') {
        const isObjectPixel = (x, y) => mask[y * targetW + x] === 1;

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (let y = 1; y < targetH - 1; y++) {
          for (let x = 1; x < targetW - 1; x++) {
            if (isObjectPixel(x, y)) {
               const isEdge = !isObjectPixel(x+1, y) || !isObjectPixel(x-1, y) || !isObjectPixel(x, y+1) || !isObjectPixel(x, y-1);
               if (isEdge) {
                   if (x % scanStep === 0 && y % scanStep === 0) {
                     const mmX = (x / targetW) * paperWidth;
                     const mmY = ((targetH - y) / targetH) * paperHeight;
                     points.push({ x: mmX, y: mmY });
                     if(mmX < minX) minX = mmX; if(mmX > maxX) maxX = mmX;
                     if(mmY < minY) minY = mmY; if(mmY > maxY) maxY = mmY;
                   }
               }
            }
          }
        }

        if(points.length > 0) {
            points.sort((a, b) => Math.atan2(a.y - ((minY+maxY)/2), a.x - ((minX+maxX)/2)) - Math.atan2(b.y - ((minY+maxY)/2), b.x - ((minX+maxX)/2)));
            
            let finalPoints = points;
            let detected = null;

            if (smartRefine) {
                const circleFit = ShapeFitter.fitCircle(points);
                if (circleFit) {
                    finalPoints = ShapeFitter.generateCircle(circleFit.cx, circleFit.cy, circleFit.r);
                    detected = 'circle';
                    minX = circleFit.cx - circleFit.r; maxX = circleFit.cx + circleFit.r;
                    minY = circleFit.cy - circleFit.r; maxY = circleFit.cy + circleFit.r;
                } else {
                    finalPoints = VectorUtils.simplify(finalPoints, 0.5); 
                    finalPoints = VectorUtils.smooth(finalPoints, curveSmoothing);
                    detected = 'poly';
                }
            } else {
                finalPoints.push(finalPoints[0]); 
            }

            setDetectedShapeType(detected);
            setObjectDims({ width: (maxX - minX), height: (maxY - minY), minX, maxX, minY, maxY });
            setProcessedPath(finalPoints);
        } else {
            setObjectDims(null); setProcessedPath([]); setDetectedShapeType(null);
        }
    } else {
        setProcessedPath([]); setObjectDims(null); setDetectedShapeType(null);
    }

    const endTime = performance.now();
    setDebugStats({ mappedPixels: mappedPixelCount, matrixValid: !isNaN(matrix[0]), processingTime: (endTime - startTime).toFixed(1) });
    
    if (segmentMode === 'auto' && (Math.abs(refR - calculatedRefColor.r) > 1 || Math.abs(refG - calculatedRefColor.g) > 1)) {
        setCalculatedRefColor({r: refR, g: refG, b: refB});
    }
    
    setIsProcessing(false);

  }, [imageSrc, corners, paperWidth, paperHeight, threshold, scanStep, curveSmoothing, imgDims, segmentMode, targetColor, selectionBox, showMask, invertResult, viewMode, calculatedRefColor, smartRefine, shadowRemoval, noiseFilter]);

  useEffect(() => {
    if (step === 'process') {
      const timer = setTimeout(processImage, 50); 
      return () => clearTimeout(timer);
    }
  }, [step, threshold, scanStep, curveSmoothing, processImage, segmentMode, showMask, invertResult, viewMode, smartRefine, shadowRemoval, noiseFilter]);

  const downloadDXF = () => {
    if (processedPath.length < 2) return;
    let dxf = "0\nSECTION\n2\nENTITIES\n0\nLWPOLYLINE\n8\nObjectLayer\n90\n" + processedPath.length + "\n70\n1\n";
    processedPath.forEach(p => { dxf += "10\n" + p.x.toFixed(3) + "\n20\n" + p.y.toFixed(3) + "\n"; });
    dxf += "0\nENDSEC\n0\nEOF";
    const blob = new Blob([dxf], { type: 'application/dxf' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'scan_shape.dxf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // --- NEW: Export Image for CAD ---
  const downloadImage = () => {
      // Create a temporary canvas at full resolution
      if (!sourcePixelData.current || !unwarpedBufferRef.current) return;
      const { width, height, data } = unwarpedBufferRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(width, height);
      // Copy raw unwarped RGB data to image
      for(let i=0; i<data.length; i++) imgData.data[i] = data[i];
      ctx.putImageData(imgData, 0, 0);
      
      const link = document.createElement('a');
      // Include physical dimensions in filename for CAD reference
      const filename = `scan_w${paperWidth}mm_h${paperHeight}mm.png`;
      link.download = filename;
      link.href = canvas.toDataURL('image/png');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
  };

  // --- AI ANALYSIS ---
  const handleAnalyzeShape = async () => {
      if (!processCanvasRef.current || !unwarpedBufferRef.current) return;
      setIsAiLoading(true);
      setShowAiPanel(true);
      setAiResult(null);

      // Create a temporary canvas to get the processed image (black on white contour) for AI
      // We want to send the "Contour" view preferably, or the Processed view.
      // Let's manually reconstruct a clean high-contrast image for the AI from the path
      const { width, height } = unwarpedBufferRef.current;
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = width; tempCanvas.height = height;
      const ctx = tempCanvas.getContext('2d');
      
      // Draw white background
      ctx.fillStyle = 'white';
      ctx.fillRect(0,0,width,height);
      
      // Draw shape (Y-axis must be inverted to match canvas coordinates)
      if (processedPath.length > 2) {
          ctx.beginPath();
          ctx.moveTo(processedPath[0].x / paperWidth * width, (1 - processedPath[0].y / paperHeight) * height);
          for(let i=1; i<processedPath.length; i++) {
              ctx.lineTo(processedPath[i].x / paperWidth * width, (1 - processedPath[i].y / paperHeight) * height);
          }
          ctx.closePath();
          ctx.lineWidth = 5;
          ctx.strokeStyle = 'black';
          ctx.stroke();
          ctx.fillStyle = '#ddd';
          ctx.fill();
      }

      // Convert to base64
      const base64Data = tempCanvas.toDataURL('image/png').split(',')[1];
      
      const result = await callGeminiVision(base64Data, paperWidth, paperHeight);
      setAiResult(result);
      setIsAiLoading(false);
  };

  if (showSplash) {
      return <SplashScreen onFinish={() => setShowSplash(false)} />;
  }

  return (
    <div className="fixed inset-0 bg-black text-neutral-200 font-sans overflow-hidden touch-none select-none h-[100dvh] flex flex-col">
      
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-neutral-900 border-b border-neutral-800 z-10 shrink-0">
        <div className="flex items-center gap-2">
           <RefreshCcw className="text-blue-500" size={20} onClick={() => setStep('capture')}/>
           <h1 className="font-bold text-lg tracking-tight text-white">ShapeScanner</h1>
        </div>
        <div className="flex items-center gap-2">
            <button 
                onClick={() => setShowDebug(!showDebug)} 
                className={`p-1 rounded ${showDebug ? 'bg-red-500 text-white' : 'text-neutral-600 hover:text-white'}`}
                title="Debug Info"
            >
                <Bug size={16} />
            </button>
            <div className="text-xs font-mono bg-neutral-800 px-2 py-1 rounded text-neutral-400">
                {step.toUpperCase()}
            </div>
        </div>
      </div>

      <div className="flex-1 relative bg-black overflow-hidden w-full h-full flex flex-col">
        
        {/* DEBUG OVERLAY */}
        {showDebug && (
            <div className="absolute top-0 left-0 bg-black/90 text-green-400 p-2 text-[10px] font-mono z-50 pointer-events-none border border-neutral-800 m-2 rounded">
                <p>Img: {imgDims.w}x{imgDims.h}</p>
                <p>Target: {Math.floor(paperWidth*2)}x{Math.floor(paperHeight*2)}</p>
                <p>Mapped: {debugStats.mappedPixels} px</p>
                <p>Time: {debugStats.processingTime}ms</p>
                <p>Mode: {viewMode}</p>
                <p>Shape: {detectedShapeType || 'None'}</p>
            </div>
        )}

        {/* STEP 1: CAPTURE */}
        {step === 'capture' && (
          <div className="h-full flex flex-col items-center justify-center p-6 space-y-8 bg-black w-full">
            <div className="text-center space-y-4">
              <div className="w-24 h-32 border-2 border-dashed border-neutral-700 mx-auto bg-neutral-900 rounded-lg flex items-center justify-center shadow-[0_0_20px_rgba(0,0,0,0.5)]">
                 <div className="w-10 h-10 bg-neutral-800 rounded-full flex items-center justify-center">
                    <ScanLine size={20} className="text-neutral-500"/>
                 </div>
              </div>
              <div className="space-y-1">
                  <h2 className="text-white font-bold text-lg">Scan Object</h2>
                  <p className="text-neutral-500 text-sm max-w-[220px] mx-auto">
                    Place object on a white sheet. Ensure all 4 corners are visible.
                  </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 w-full max-w-xs">
                <button onClick={() => fileInputRef.current.click()} className="flex flex-col items-center justify-center bg-blue-600 hover:bg-blue-500 active:scale-95 transition-all p-6 rounded-2xl shadow-lg group">
                    <Upload size={32} className="mb-2 group-hover:-translate-y-1 transition-transform"/><span className="font-bold">Upload</span>
                </button>
                <button onClick={() => fileInputRef.current.click()} className="flex flex-col items-center justify-center bg-neutral-800 hover:bg-neutral-700 active:scale-95 transition-all p-6 rounded-2xl border border-neutral-700 group">
                    <Camera size={32} className="mb-2 group-hover:-translate-y-1 transition-transform"/><span className="font-bold">Camera</span>
                </button>
            </div>
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
          </div>
        )}

        {/* STEP 2: CALIBRATE */}
        {step === 'calibrate' && (
        <div className="w-full h-full relative flex flex-col">
            <div className="flex-1 relative overflow-hidden">
                <canvas 
                    ref={canvasRef}
                    className="w-full h-full block cursor-crosshair touch-none bg-black"
                    onMouseDown={(e) => handleStart(e.clientX, e.clientY)}
                    onMouseMove={(e) => handleMove(e.clientX, e.clientY)}
                    onMouseUp={handleEnd}
                    onMouseLeave={handleEnd}
                    onTouchStart={(e) => { const t = e.touches[0]; handleStart(t.clientX, t.clientY); }}
                    onTouchMove={(e) => { const t = e.touches[0]; handleMove(t.clientX, t.clientY); }}
                    onTouchEnd={handleEnd}
                    onWheel={handleWheel}
                />

                <div className="absolute top-4 right-4 flex flex-col gap-2 pointer-events-none">
                <div className="bg-black/80 backdrop-blur rounded-lg p-1 pointer-events-auto shadow-lg flex flex-col gap-1 border border-neutral-800">
                    <button onClick={() => setView(v => ({...v, scale: v.scale * 1.2}))} className="p-2 hover:bg-neutral-800 rounded text-neutral-300"><ZoomIn size={20}/></button>
                    <button onClick={() => setView(v => ({...v, scale: v.scale * 0.8}))} className="p-2 hover:bg-neutral-800 rounded text-neutral-300"><ZoomOut size={20}/></button>
                    <button onClick={fitToScreen} className="p-2 hover:bg-neutral-800 rounded text-blue-500"><Maximize2 size={20}/></button>
                </div>
                </div>
                
                <div className="absolute top-4 left-4 pointer-events-none flex flex-col gap-4">
                    <button 
                        onClick={autoDetectCorners}
                        className="pointer-events-auto bg-blue-600/90 hover:bg-blue-600 backdrop-blur text-white px-4 py-2 rounded-full text-xs font-bold shadow-lg flex items-center gap-2 w-max transition-colors"
                    >
                        <ScanLine size={14} /> Auto-Detect
                    </button>

                    <div className="bg-black/80 backdrop-blur rounded-lg p-3 pointer-events-auto shadow-lg flex flex-col gap-3 border border-neutral-800 w-40">
                        <div className="flex items-center justify-between">
                            <span className="text-xs font-bold text-neutral-400 flex items-center gap-2"><Palette size={12}/> B&W</span>
                            <button 
                                onClick={() => setCalMonochrome(!calMonochrome)}
                                className={`w-8 h-4 rounded-full relative transition-colors ${calMonochrome ? 'bg-blue-600' : 'bg-neutral-700'}`}
                            >
                                <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${calMonochrome ? 'left-4.5' : 'left-0.5'}`} />
                            </button>
                        </div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs text-neutral-400">
                                <span className="flex items-center gap-1"><Sun size={12} /> Contrast</span>
                                <span>{calContrast}%</span>
                            </div>
                            <input 
                                type="range" min="50" max="200" value={calContrast} 
                                onChange={(e) => setCalContrast(Number(e.target.value))}
                                className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                        </div>
                    </div>
                </div>
            </div>

            <div className="w-full bg-neutral-900 p-4 rounded-t-2xl shadow-[0_-4px_20px_rgba(0,0,0,0.5)] border-t border-neutral-800 shrink-0 z-20">
                <div className="flex gap-3 mb-4">
                    <div className="flex flex-col gap-1 w-24 shrink-0">
                        <label className="text-[10px] text-neutral-500 uppercase font-bold flex justify-between items-center mb-1">
                            Format
                            <button onClick={toggleOrientation} className="text-neutral-400 hover:text-white transition-colors bg-neutral-800 p-1 rounded"><RotateCcw size={10} /></button>
                        </label>
                        <div className="flex flex-col gap-1.5">
                            <button onClick={() => applyPreset(210, 297)} className="bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 px-2 py-1.5 rounded text-xs flex items-center gap-2 text-neutral-300 transition-colors"><FileText size={12}/> A4</button>
                            <button onClick={() => applyPreset(215.9, 279.4)} className="bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 px-2 py-1.5 rounded text-xs flex items-center gap-2 text-neutral-300 transition-colors"><FileText size={12}/> Letter</button>
                            <button onClick={() => applyPreset(85.6, 53.98)} className="bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 px-2 py-1.5 rounded text-xs flex items-center gap-2 text-neutral-300 transition-colors"><CreditCard size={12}/> Card</button>
                        </div>
                    </div>

                    <div className="flex-1 flex gap-3">
                        <div className="flex-1">
                            <label className="text-[10px] text-neutral-500 uppercase font-bold mb-1 block">Width (mm)</label>
                            <input type="number" value={paperWidth} onChange={e => setPaperWidth(Number(e.target.value))} className="w-full bg-black border border-neutral-700 rounded-lg p-3 text-white font-mono text-sm focus:border-blue-500 focus:outline-none transition-colors"/>
                        </div>
                        <div className="flex-1">
                            <label className="text-[10px] text-neutral-500 uppercase font-bold mb-1 block">Height (mm)</label>
                            <input type="number" value={paperHeight} onChange={e => setPaperHeight(Number(e.target.value))} className="w-full bg-black border border-neutral-700 rounded-lg p-3 text-white font-mono text-sm focus:border-blue-500 focus:outline-none transition-colors"/>
                        </div>
                    </div>
                </div>
                <button onClick={() => setStep('process')} className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3.5 rounded-xl flex items-center justify-center gap-2 transition-all active:scale-[0.98] shadow-lg shadow-blue-900/20">
                    <Check size={18} /> Confirm Dimensions
                </button>
            </div>
        </div>
        )}

        {/* STEP 3: PROCESS */}
        {step === 'process' && (
        <div className="flex flex-col h-full w-full relative">
            
            {/* AI ANALYST SIDE PANEL/MODAL */}
            {showAiPanel && (
                <div className="absolute inset-0 z-[60] bg-black/50 backdrop-blur-sm flex justify-end">
                    <div className="w-full max-w-sm h-full bg-neutral-900 border-l border-neutral-800 shadow-2xl flex flex-col animate-[slideInRight_0.3s_ease-out]">
                        <div className="flex items-center justify-between p-4 border-b border-neutral-800">
                            <h3 className="font-bold text-white flex items-center gap-2">
                                <Sparkles size={18} className="text-purple-500" /> AI Analyst
                            </h3>
                            <button onClick={() => setShowAiPanel(false)} className="p-2 hover:bg-neutral-800 rounded text-neutral-400">
                                <X size={20} />
                            </button>
                        </div>
                        <div className="flex-1 p-6 overflow-y-auto">
                            {isAiLoading ? (
                                <div className="flex flex-col items-center justify-center h-full space-y-4">
                                    <div className="w-12 h-12 border-4 border-purple-600/30 border-t-purple-600 rounded-full animate-spin"></div>
                                    <p className="text-neutral-400 text-sm animate-pulse">Analyzing geometry...</p>
                                </div>
                            ) : (
                                <div className="space-y-6">
                                    <div className="bg-neutral-800/50 p-4 rounded-xl border border-neutral-700/50">
                                        <h4 className="text-xs font-bold text-neutral-500 uppercase mb-2">Detected Part</h4>
                                        <div className="text-white text-sm leading-relaxed whitespace-pre-line">{aiResult}</div>
                                    </div>
                                    <div className="text-xs text-neutral-500 text-center">
                                        AI analysis based on {processedPath.length} vector points.
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            <div className="flex-1 min-h-0 flex flex-col w-full bg-black relative">
                {/* Process Toolbar */}
                <div className="w-full text-center text-xs text-neutral-400 p-2 flex justify-between items-center flex-wrap gap-2 shrink-0 z-10">
                    <div className="flex items-center gap-2">
                         <div className="flex bg-neutral-900 rounded-lg p-1 border border-neutral-800">
                             <button 
                                onClick={() => setViewMode('original')}
                                className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 text-[10px] uppercase font-bold transition-all ${viewMode === 'original' ? 'bg-neutral-700 text-white shadow-sm' : 'text-neutral-500 hover:text-neutral-300'}`}
                             >
                                <ImageIcon size={12}/> Orig
                             </button>
                             <button 
                                onClick={() => setViewMode('heatmap')}
                                className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 text-[10px] uppercase font-bold transition-all ${viewMode === 'heatmap' ? 'bg-orange-900/80 text-orange-200 shadow-sm' : 'text-neutral-500 hover:text-neutral-300'}`}
                             >
                                <Flame size={12}/> Heat
                             </button>
                             <button 
                                onClick={() => setViewMode('processed')}
                                className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 text-[10px] uppercase font-bold transition-all ${viewMode === 'processed' ? 'bg-emerald-900/80 text-emerald-200 shadow-sm' : 'text-neutral-500 hover:text-neutral-300'}`}
                             >
                                <Layers size={12}/> Proc
                             </button>
                             <button 
                                onClick={() => setViewMode('contour')}
                                className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 text-[10px] uppercase font-bold transition-all ${viewMode === 'contour' ? 'bg-purple-900/80 text-purple-200 shadow-sm' : 'text-neutral-500 hover:text-neutral-300'}`}
                             >
                                <PenTool size={12}/> Cont
                             </button>
                         </div>
                    </div>

                    <div className="flex gap-2">
                         <button 
                            onClick={handleAnalyzeShape}
                            disabled={processedPath.length < 3}
                            className="px-3 py-1.5 rounded-lg flex items-center gap-2 border text-[10px] font-bold uppercase tracking-wider transition-all bg-purple-600/20 text-purple-400 border-purple-500/50 hover:bg-purple-600/30 disabled:opacity-50 disabled:cursor-not-allowed"
                         >
                            <Sparkles size={12}/> AI
                         </button>
                         <button 
                            onClick={() => setIsPicking(!isPicking)} 
                            className={`px-3 py-1.5 rounded-lg flex items-center gap-2 border text-[10px] font-bold uppercase tracking-wider transition-all ${
                                isPicking 
                                ? 'bg-amber-500 text-black border-amber-400 shadow-[0_0_10px_rgba(245,158,11,0.4)] animate-pulse' 
                                : segmentMode === 'manual-bg' 
                                    ? 'bg-amber-900/30 text-amber-500 border-amber-900' 
                                    : 'bg-neutral-800 border-neutral-700 text-neutral-400 hover:bg-neutral-700'
                            }`}
                         >
                            <Pipette size={12}/> {isPicking ? 'Set' : 'Ref'}
                        </button>
                    </div>
                </div>
                
                {/* Main Canvas Area - Centered and Scaled */}
                <div className="flex-1 w-full relative flex items-center justify-center overflow-hidden p-2">
                    <div 
                        className={`relative border border-neutral-800 shadow-2xl transition-colors duration-300 ${isPicking ? 'cursor-crosshair ring-2 ring-amber-500/50' : ''}`}
                        style={{
                            aspectRatio: `${paperWidth}/${paperHeight}`,
                            maxHeight: '100%',
                            maxWidth: '100%',
                            backgroundColor: viewMode === 'contour' ? 'white' : '#111', 
                            backgroundImage: viewMode === 'contour' ? 'none' : 'radial-gradient(#333 1px, transparent 1px)',
                            backgroundSize: '20px 20px'
                        }}
                    >
                        {!sourcePixelData.current && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/90 text-white flex-col z-50">
                                <AlertTriangle className="text-amber-500 mb-3" size={48} />
                                <span className="text-xl font-bold mb-1">Session Expired</span>
                                <button onClick={() => setStep('capture')} className="px-6 py-2 bg-blue-600 rounded-full font-bold text-sm mt-4 hover:bg-blue-500 transition-colors">Reload Image</button>
                            </div>
                        )}
                        <canvas 
                            ref={processCanvasRef} 
                            className="w-full h-full object-contain touch-none relative z-10"
                            onMouseDown={handleProcessStart}
                            onMouseMove={handleProcessMove}
                            onMouseUp={handleProcessEnd}
                            onTouchStart={(e) => { const t = e.touches[0]; handleProcessStart({ clientX: t.clientX, clientY: t.clientY }); }}
                            onTouchMove={(e) => { const t = e.touches[0]; handleProcessMove({ clientX: t.clientX, clientY: t.clientY }); }}
                            onTouchEnd={handleProcessEnd}
                        />
                        
                        {(viewMode === 'processed' || viewMode === 'contour') && (
                            <svg className="absolute top-0 left-0 w-full h-full pointer-events-none z-20" viewBox={`0 0 100 100`} preserveAspectRatio="none">
                                {processedPath.length > 0 && (
                                    <>
                                        <path 
                                            d={`M ${processedPath.map(p => `${(p.x/paperWidth)*100} ${(1 - p.y/paperHeight)*100}`).join(" L ")} Z`}
                                            fill="none" 
                                            stroke={viewMode === 'contour' ? '#000000' : '#10b981'} 
                                            strokeWidth={viewMode === 'contour' ? "1.5" : "1"}
                                            vectorEffect="non-scaling-stroke"
                                        />
                                        {objectDims && viewMode !== 'heatmap' && (
                                            <g>
                                                {/* Smart Guide Lines */}
                                                {viewMode !== 'contour' && (
                                                <rect 
                                                    x={`${(objectDims.minX/paperWidth)*100}%`} 
                                                    y={`${(1 - objectDims.maxY/paperHeight)*100}%`}
                                                    width={`${(objectDims.width/paperWidth)*100}%`}
                                                    height={`${(objectDims.height/paperHeight)*100}%`}
                                                    fill="none" stroke="rgba(255,255,255,0.3)" strokeDasharray="2" vectorEffect="non-scaling-stroke"
                                                />
                                                )}
                                                
                                                {/* Measurements */}
                                                <line 
                                                    x1={`${(objectDims.minX/paperWidth)*100}%`} y1={`${(1 - objectDims.maxY/paperHeight)*100}%`}
                                                    x2={`${(objectDims.maxX/paperWidth)*100}%`} y2={`${(1 - objectDims.maxY/paperHeight)*100}%`}
                                                    stroke={viewMode === 'contour' ? 'black' : 'white'} strokeWidth="0.5" transform="translate(0, -5)" vectorEffect="non-scaling-stroke"
                                                />
                                                <text 
                                                    x={`${((objectDims.minX + objectDims.width/2)/paperWidth)*100}%`} 
                                                    y={`${(1 - objectDims.maxY/paperHeight)*100}%`} 
                                                    dy="-8" 
                                                    fill={viewMode === 'contour' ? 'black' : 'white'} 
                                                    textAnchor="middle" fontSize="3" fontWeight="bold" 
                                                    style={{textShadow: viewMode === 'contour' ? 'none' : '0px 2px 4px rgba(0,0,0,1)'}}
                                                >
                                                    {objectDims.width.toFixed(1)} mm
                                                </text>

                                                <line 
                                                    x1={`${(objectDims.maxX/paperWidth)*100}%`} y1={`${(1 - objectDims.maxY/paperHeight)*100}%`}
                                                    x2={`${(objectDims.maxX/paperWidth)*100}%`} y2={`${(1 - objectDims.minY/paperHeight)*100}%`}
                                                    stroke={viewMode === 'contour' ? 'black' : 'white'} strokeWidth="0.5" transform="translate(5, 0)" vectorEffect="non-scaling-stroke"
                                                />
                                                <text 
                                                    x={`${(objectDims.maxX/paperWidth)*100}%`} 
                                                    y={`${(1 - (objectDims.minY + objectDims.height/2)/paperHeight)*100}%`} 
                                                    dx="8" 
                                                    fill={viewMode === 'contour' ? 'black' : 'white'} 
                                                    dominantBaseline="middle" fontSize="3" fontWeight="bold"
                                                    style={{textShadow: viewMode === 'contour' ? 'none' : '0px 2px 4px rgba(0,0,0,1)'}}
                                                >
                                                    {objectDims.height.toFixed(1)} mm
                                                </text>
                                            </g>
                                        )}
                                    </>
                                )}
                            </svg>
                        )}
                        
                        {isPicking && (
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-amber-500/90 backdrop-blur text-black font-bold px-6 py-3 rounded-full shadow-2xl pointer-events-none text-sm animate-bounce z-50 border-2 border-white">
                                Tap Background Color
                            </div>
                        )}
                        
                        {detectedShapeType && (
                            <div className="absolute bottom-4 left-4 bg-green-600/90 backdrop-blur text-white px-3 py-1.5 rounded-full shadow-lg text-[10px] font-bold uppercase tracking-widest z-30 flex items-center gap-1.5 border border-green-500">
                                <Check size={12} className="stroke-[3]" /> {detectedShapeType} Detected
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="bg-neutral-900 p-5 rounded-t-3xl border-t border-neutral-800 space-y-5 z-20 shrink-0 shadow-[0_-10px_40px_rgba(0,0,0,0.5)] max-h-[40vh] overflow-y-auto">
                {viewMode !== 'original' ? (
                    <>
                        <div className="flex items-center gap-2 mb-2 justify-between">
                            <div className="flex items-center gap-2">
                                <Settings size={18} className="text-neutral-400"/>
                                <div className="flex flex-col leading-none">
                                    <span className="font-bold text-sm text-white">Detection</span>
                                    <span className="text-[10px] text-neutral-500 uppercase font-bold mt-0.5">
                                        {segmentMode === 'auto' && 'Auto Contrast'}
                                        {segmentMode === 'manual-bg' && 'Background Ref'}
                                        {segmentMode === 'manual-obj' && 'Object Ref'}
                                    </span>
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <button 
                                    onClick={() => setSmartRefine(!smartRefine)}
                                    className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 border text-[10px] font-bold uppercase tracking-wider transition-all ${smartRefine ? 'bg-blue-600 border-blue-500 text-white shadow-[0_0_10px_rgba(37,99,235,0.3)]' : 'bg-neutral-800 border-neutral-700 text-neutral-400'}`}
                                >
                                    <PenTool size={12}/> Smart
                                </button>
                                <button 
                                    onClick={() => setInvertResult(!invertResult)}
                                    className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 border text-[10px] font-bold uppercase tracking-wider transition-all ${invertResult ? 'bg-purple-600 border-purple-500 text-white shadow-[0_0_10px_rgba(147,51,234,0.3)]' : 'bg-neutral-800 border-neutral-700 text-neutral-400'}`}
                                >
                                    {invertResult ? <ToggleRight size={14}/> : <ToggleLeft size={14}/>} Invert
                                </button>
                                <div className="flex items-center gap-2 text-xs text-neutral-400 pl-2 border-l border-neutral-800">
                                    <div 
                                        className="w-6 h-6 rounded-full border border-neutral-600 shadow-inner" 
                                        style={{backgroundColor: `rgb(${calculatedRefColor.r},${calculatedRefColor.g},${calculatedRefColor.b})`}} 
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-1.5">
                                <div className="flex justify-between text-[10px] uppercase font-bold text-neutral-400 tracking-wider"><span>Threshold Sensitivity</span><span className="text-white">{threshold}</span></div>
                                <input type="range" min="1" max="150" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                            </div>
                            
                            <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-[10px] uppercase font-bold text-neutral-400 tracking-wider"><span>Shadow Removal</span><span className="text-white">{shadowRemoval}</span></div>
                                    <input type="range" min="0" max="10" value={shadowRemoval} onChange={(e) => setShadowRemoval(Number(e.target.value))} className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                                </div>
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-[10px] uppercase font-bold text-neutral-400 tracking-wider"><span>Curve Smooth</span><span className="text-white">{curveSmoothing}</span></div>
                                    <input type="range" min="0" max="5" value={curveSmoothing} onChange={(e) => setCurveSmoothing(Number(e.target.value))} className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                                </div>
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-[10px] uppercase font-bold text-neutral-400 tracking-wider"><span>Detail Scan</span><span className="text-white">{scanStep}px</span></div>
                                    <input type="range" min="1" max="10" value={scanStep} onChange={(e) => setScanStep(Number(e.target.value))} className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                                </div>
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-[10px] uppercase font-bold text-neutral-400 tracking-wider"><span>Noise Filter</span><span className="text-white">{noiseFilter}px</span></div>
                                    <input type="range" min="0" max="10" value={noiseFilter} onChange={(e) => setNoiseFilter(Number(e.target.value))} className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                                </div>
                            </div>
                        </div>

                        <div className="flex gap-3 mt-2 pb-6">
                            <button onClick={downloadImage} className="flex-1 bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 text-white font-bold py-3.5 rounded-xl flex items-center justify-center gap-2 text-xs transition-all active:scale-[0.98]">
                                <ImageIcon size={16} /> Save Image
                            </button>
                            <button onClick={downloadDXF} disabled={processedPath.length < 3} className="flex-1 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-3.5 rounded-xl flex items-center justify-center gap-2 text-xs transition-all active:scale-[0.98] shadow-lg shadow-emerald-900/20">
                                <Download size={16} /> Save Vector DXF
                            </button>
                        </div>
                    </>
                ) : (
                    <div className="text-center py-8 text-neutral-500 text-sm flex flex-col gap-4 items-center pb-12">
                        <p className="max-w-[200px]">Switch to <b>Processed</b> or <b>Heatmap</b> mode above to configure detection settings.</p>
                        <button onClick={downloadImage} className="w-full max-w-xs bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 text-white font-bold py-3.5 rounded-xl flex items-center justify-center gap-2 text-xs transition-all">
                            <ImageIcon size={16} /> Save Original Image
                        </button>
                    </div>
                )}
            </div>
        </div>
        )}
      </div>
    </div>
  );
};

export default ShapeScanner;