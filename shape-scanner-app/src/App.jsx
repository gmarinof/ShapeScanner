import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera as CameraIcon, Upload, Check, RefreshCcw, Settings, Download, ScanLine, ZoomIn, ZoomOut, Maximize2, MousePointer2, Eye, EyeOff, Sun, Palette, Pipette, ToggleLeft, ToggleRight, AlertTriangle, Image as ImageIcon, Layers, Flame, Bug, PenTool, FileText, CreditCard, BoxSelect, Eraser, RotateCcw, Sparkles, X, Move, ChevronDown, ChevronUp, HelpCircle, ChevronLeft, ChevronRight, BookOpen, Circle } from 'lucide-react';
import { Capacitor } from '@capacitor/core';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import { Filesystem, Directory, Encoding } from '@capacitor/filesystem';

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
 * CONTOUR TRACER - For multi-polygon detection with holes using boundary tracing
 */
class ContourTracer {
  // 4-connectivity directions (N4): right, down, left, up
  static DIRS4 = [[1,0], [0,1], [-1,0], [0,-1]];
  // 8-connectivity directions (N8): clockwise starting from right
  static DIRS8 = [[1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]];

  // Connected components labeling using flood-fill (4-connectivity)
  static labelComponents(mask, width, height) {
    const labels = new Int32Array(width * height);
    let currentLabel = 0;
    const components = [];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (mask[idx] === 1 && labels[idx] === 0) {
          currentLabel++;
          const pixels = [];
          const queue = [[x, y]];
          let minX = x, maxX = x, minY = y, maxY = y;
          
          while (queue.length > 0) {
            const [cx, cy] = queue.shift();
            const cidx = cy * width + cx;
            if (cx < 0 || cx >= width || cy < 0 || cy >= height) continue;
            if (mask[cidx] !== 1 || labels[cidx] !== 0) continue;
            
            labels[cidx] = currentLabel;
            pixels.push({ x: cx, y: cy });
            if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
            
            // 4-connectivity only
            queue.push([cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]);
          }
          
          if (pixels.length > 20) {
            components.push({ label: currentLabel, pixels, size: pixels.length, bbox: {minX, maxX, minY, maxY} });
          }
        }
      }
    }
    return { labels, components };
  }

  // 4-connectivity boundary tracing - prevents diagonal leakage
  static traceBoundary4(mask, width, height, startX, startY, componentLabel, labels) {
    const contour = [];
    const isInside = (x, y) => {
      if (x < 0 || x >= width || y < 0 || y >= height) return false;
      if (mask[y * width + x] !== 1) return false;
      if (componentLabel !== null && labels !== null) {
        return labels[y * width + x] === componentLabel;
      }
      return true;
    };
    
    if (!isInside(startX, startY)) return contour;
    
    // Find first boundary pixel (leftmost on starting row)
    let sx = startX, sy = startY;
    while (sx > 0 && isInside(sx - 1, sy)) sx--;
    
    let x = sx, y = sy;
    let dir = 3; // Start looking from up direction
    const maxIter = width * height * 2;
    let iter = 0;
    
    do {
      contour.push({ x, y });
      // Search for next boundary pixel in 4-neighborhood (counterclockwise from backtrack)
      let found = false;
      for (let i = 0; i < 4; i++) {
        const checkDir = (dir + 3 + i) % 4; // Start from dir+3 (turn left from incoming)
        const nx = x + this.DIRS4[checkDir][0];
        const ny = y + this.DIRS4[checkDir][1];
        if (isInside(nx, ny)) {
          x = nx; y = ny;
          dir = checkDir;
          found = true;
          break;
        }
      }
      if (!found) break;
      iter++;
    } while ((x !== sx || y !== sy) && iter < maxIter);
    
    return contour;
  }

  // Find holes by detecting enclosed background regions - with proper component ownership check
  static findHoles(mask, labels, componentLabel, width, height, bbox) {
    const holes = [];
    const { minX, maxX, minY, maxY } = bbox;
    const visited = new Uint8Array(width * height);
    
    // Mark all background connected to image border as external
    const externalQueue = [];
    for (let x = 0; x < width; x++) {
      if (mask[x] === 0 && visited[x] === 0) { externalQueue.push([x, 0]); visited[x] = 2; }
      const bottomIdx = (height-1) * width + x;
      if (mask[bottomIdx] === 0 && visited[bottomIdx] === 0) { externalQueue.push([x, height-1]); visited[bottomIdx] = 2; }
    }
    for (let y = 0; y < height; y++) {
      const leftIdx = y * width;
      const rightIdx = y * width + width - 1;
      if (mask[leftIdx] === 0 && visited[leftIdx] === 0) { externalQueue.push([0, y]); visited[leftIdx] = 2; }
      if (mask[rightIdx] === 0 && visited[rightIdx] === 0) { externalQueue.push([width-1, y]); visited[rightIdx] = 2; }
    }
    
    // Flood-fill external background (4-connectivity)
    while (externalQueue.length > 0) {
      const [cx, cy] = externalQueue.shift();
      for (const [dx, dy] of this.DIRS4) {
        const nx = cx + dx, ny = cy + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const nidx = ny * width + nx;
        if (mask[nidx] === 0 && visited[nidx] === 0) {
          visited[nidx] = 2;
          externalQueue.push([nx, ny]);
        }
      }
    }
    
    // Find internal background regions (holes) within component bounding box
    for (let y = minY + 1; y < maxY; y++) {
      for (let x = minX + 1; x < maxX; x++) {
        const idx = y * width + x;
        if (mask[idx] === 0 && visited[idx] === 0) {
          const holePixels = [];
          const boundaryLabels = new Set();
          const queue = [[x, y]];
          let isEnclosed = true;
          
          while (queue.length > 0) {
            const [cx, cy] = queue.shift();
            const cidx = cy * width + cx;
            if (cx < 0 || cx >= width || cy < 0 || cy >= height) { isEnclosed = false; continue; }
            if (visited[cidx] !== 0) { if (visited[cidx] === 2) isEnclosed = false; continue; }
            if (mask[cidx] === 1) continue; // Skip foreground pixels
            
            visited[cidx] = 1;
            holePixels.push({ x: cx, y: cy });
            
            // Check all 4 neighbors and collect boundary labels from foreground neighbors
            for (const [dx, dy] of this.DIRS4) {
              const nx = cx + dx, ny = cy + dy;
              if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
              const nidx = ny * width + nx;
              if (mask[nidx] === 1) {
                // This neighbor is foreground - record its component label
                if (labels[nidx] !== 0) boundaryLabels.add(labels[nidx]);
              } else if (visited[nidx] === 0) {
                // This neighbor is background and unvisited - add to queue
                queue.push([nx, ny]);
              }
            }
          }
          
          // Accept hole if enclosed AND either: 
          // 1. Bordered exclusively by the target component, OR
          // 2. Has no boundary labels but is enclosed (fully interior hole)
          const isOwnedByComponent = boundaryLabels.size === 0 || 
            (boundaryLabels.size === 1 && boundaryLabels.has(componentLabel));
          
          if (isEnclosed && isOwnedByComponent && holePixels.length > 10) {
            holes.push(holePixels);
          }
        }
      }
    }
    return holes;
  }

  // Simplify contour by sampling points at intervals
  static simplifyContour(contour, step = 2) {
    if (contour.length < 10) return contour;
    const result = [];
    for (let i = 0; i < contour.length; i += step) {
      result.push(contour[i]);
    }
    return result;
  }

  // Compute signed area to determine winding direction
  static signedArea(points) {
    let area = 0;
    const n = points.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += points[i].x * points[j].y;
      area -= points[j].x * points[i].y;
    }
    return area / 2;
  }

  // Reverse winding direction of a contour
  static reverseContour(contour) {
    return contour.slice().reverse();
  }

  // Main function to detect all polygons with holes
  static detectPolygons(mask, width, height, paperWidth, paperHeight, scanStep = 2) {
    const { labels, components } = this.labelComponents(mask, width, height);
    const polygons = [];

    components.sort((a, b) => b.size - a.size);

    for (const component of components) {
      // Find starting point (topmost, then leftmost pixel)
      let startPixel = component.pixels[0];
      for (const p of component.pixels) {
        if (p.y < startPixel.y || (p.y === startPixel.y && p.x < startPixel.x)) {
          startPixel = p;
        }
      }
      
      // Trace outer boundary using 4-connectivity
      const boundary = this.traceBoundary4(mask, width, height, startPixel.x, startPixel.y, component.label, labels);
      if (boundary.length < 5) continue;
      
      // Simplify the contour
      const simplified = this.simplifyContour(boundary, scanStep);
      
      // Convert to mm coordinates
      let outerContour = simplified.map(p => ({
        x: (p.x / width) * paperWidth,
        y: ((height - p.y) / height) * paperHeight
      }));
      
      // Ensure outer contour is counter-clockwise (positive area = CCW in CAD convention)
      if (this.signedArea(outerContour) < 0) {
        outerContour = this.reverseContour(outerContour);
      }
      
      // Close the contour
      if (outerContour.length > 0) {
        outerContour.push({ ...outerContour[0] });
      }

      // Find and trace holes
      const holes = this.findHoles(mask, labels, component.label, width, height, component.bbox);
      const holeContours = [];

      for (const holePixels of holes) {
        if (holePixels.length < 5) continue;
        let startHole = holePixels[0];
        for (const p of holePixels) {
          if (p.y < startHole.y || (p.y === startHole.y && p.x < startHole.x)) startHole = p;
        }
        
        // Create temporary mask for hole tracing
        const holeMask = new Uint8Array(width * height);
        holePixels.forEach(p => { holeMask[p.y * width + p.x] = 1; });
        
        // Use 4-connectivity for hole tracing to prevent diagonal leakage
        const holeBoundary = this.traceBoundary4(holeMask, width, height, startHole.x, startHole.y, null, null);
        if (holeBoundary.length < 5) continue;
        
        const holeSimplified = this.simplifyContour(holeBoundary, scanStep);
        let holeContour = holeSimplified.map(p => ({
          x: (p.x / width) * paperWidth,
          y: ((height - p.y) / height) * paperHeight
        }));
        
        // Ensure hole contour is clockwise (negative area = CW = hole in CAD convention)
        if (this.signedArea(holeContour) > 0) {
          holeContour = this.reverseContour(holeContour);
        }
        
        if (holeContour.length > 0) {
          holeContour.push({ ...holeContour[0] });
          holeContours.push(holeContour);
        }
      }

      polygons.push({
        outer: outerContour,
        holes: holeContours,
        bbox: {
          minX: (component.bbox.minX / width) * paperWidth,
          maxX: (component.bbox.maxX / width) * paperWidth,
          minY: ((height - component.bbox.maxY) / height) * paperHeight,
          maxY: ((height - component.bbox.minY) / height) * paperHeight
        },
        size: component.size
      });
    }

    return polygons;
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
 * EDGE DETECTOR - Sobel gradients + Canny-style hysteresis for corner detection
 */
class EdgeDetector {
  // Sobel kernels for gradient computation
  static SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
  static SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

  // Convert RGB image data to grayscale with optional paper color distance
  static toGrayscale(srcData, width, height, paperColor = null) {
    const gray = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const r = srcData[i], g = srcData[i + 1], b = srcData[i + 2];
        if (paperColor) {
          // Use color distance from paper color
          const dist = Math.sqrt(
            (r - paperColor.r) ** 2 + 
            (g - paperColor.g) ** 2 + 
            (b - paperColor.b) ** 2
          );
          gray[y * width + x] = dist;
        } else {
          // Standard grayscale
          gray[y * width + x] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
      }
    }
    return gray;
  }

  // Apply Gaussian blur for noise reduction (5x5 kernel) with border mirroring
  static gaussianBlur(gray, width, height) {
    const kernel = [
      [1, 4, 7, 4, 1],
      [4, 16, 26, 16, 4],
      [7, 26, 41, 26, 7],
      [4, 16, 26, 16, 4],
      [1, 4, 7, 4, 1]
    ];
    const kernelSum = 273;
    const output = new Float32Array(width * height);
    
    // Helper to get pixel with mirror boundary handling (reflect at border)
    const getPixel = (x, y) => {
      const mx = x < 0 ? -x - 1 : (x >= width ? 2 * width - 1 - x : x);
      const my = y < 0 ? -y - 1 : (y >= height ? 2 * height - 1 - y : y);
      return gray[Math.max(0, Math.min(height - 1, my)) * width + Math.max(0, Math.min(width - 1, mx))];
    };
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        for (let ky = -2; ky <= 2; ky++) {
          for (let kx = -2; kx <= 2; kx++) {
            sum += getPixel(x + kx, y + ky) * kernel[ky + 2][kx + 2];
          }
        }
        output[y * width + x] = sum / kernelSum;
      }
    }
    return output;
  }

  // Compute Sobel gradients
  static sobelGradients(gray, width, height) {
    const gx = new Float32Array(width * height);
    const gy = new Float32Array(width * height);
    const magnitude = new Float32Array(width * height);
    const direction = new Float32Array(width * height);
    let maxMag = 0;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sumX = 0, sumY = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const val = gray[(y + ky) * width + (x + kx)];
            sumX += val * this.SOBEL_X[ky + 1][kx + 1];
            sumY += val * this.SOBEL_Y[ky + 1][kx + 1];
          }
        }
        const idx = y * width + x;
        gx[idx] = sumX;
        gy[idx] = sumY;
        const mag = Math.sqrt(sumX * sumX + sumY * sumY);
        magnitude[idx] = mag;
        direction[idx] = Math.atan2(sumY, sumX);
        if (mag > maxMag) maxMag = mag;
      }
    }

    return { gx, gy, magnitude, direction, maxMag };
  }

  // Non-maximum suppression for edge thinning
  static nonMaxSuppression(magnitude, direction, width, height) {
    const output = new Float32Array(width * height);
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        const mag = magnitude[idx];
        let angle = direction[idx] * (180 / Math.PI);
        if (angle < 0) angle += 180;
        
        let n1 = 0, n2 = 0;
        // Quantize to 4 directions
        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
          // Horizontal edge, compare with left and right
          n1 = magnitude[idx - 1];
          n2 = magnitude[idx + 1];
        } else if (angle >= 22.5 && angle < 67.5) {
          // Diagonal (/), compare with upper-right and lower-left
          n1 = magnitude[(y - 1) * width + (x + 1)];
          n2 = magnitude[(y + 1) * width + (x - 1)];
        } else if (angle >= 67.5 && angle < 112.5) {
          // Vertical edge, compare with up and down
          n1 = magnitude[(y - 1) * width + x];
          n2 = magnitude[(y + 1) * width + x];
        } else {
          // Diagonal (\), compare with upper-left and lower-right
          n1 = magnitude[(y - 1) * width + (x - 1)];
          n2 = magnitude[(y + 1) * width + (x + 1)];
        }
        
        // Keep only if it's a local maximum
        if (mag >= n1 && mag >= n2) {
          output[idx] = mag;
        }
      }
    }
    
    return output;
  }

  // Double threshold and hysteresis tracking (Canny-style)
  static hysteresisThreshold(thinned, width, height, lowRatio = 0.05, highRatio = 0.15) {
    // Find max value
    let maxVal = 0;
    for (let i = 0; i < thinned.length; i++) {
      if (thinned[i] > maxVal) maxVal = thinned[i];
    }
    
    const lowThreshold = maxVal * lowRatio;
    const highThreshold = maxVal * highRatio;
    const edges = new Uint8Array(width * height);
    
    // Mark strong and weak edges
    for (let i = 0; i < thinned.length; i++) {
      if (thinned[i] >= highThreshold) {
        edges[i] = 2; // Strong edge
      } else if (thinned[i] >= lowThreshold) {
        edges[i] = 1; // Weak edge
      }
    }
    
    // Hysteresis: promote weak edges connected to strong edges
    const DIRS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
    let changed = true;
    while (changed) {
      changed = false;
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          const idx = y * width + x;
          if (edges[idx] === 1) {
            // Check if connected to a strong edge
            for (const [dy, dx] of DIRS) {
              if (edges[(y + dy) * width + (x + dx)] === 2) {
                edges[idx] = 2;
                changed = true;
                break;
              }
            }
          }
        }
      }
    }
    
    // Final edge map (only strong edges)
    const result = new Uint8Array(width * height);
    for (let i = 0; i < edges.length; i++) {
      result[i] = edges[i] === 2 ? 1 : 0;
    }
    
    return result;
  }

  // Complete Canny edge detection pipeline
  static cannyEdgeDetection(srcData, width, height, paperColor = null, lowRatio = 0.05, highRatio = 0.15) {
    // Step 1: Convert to grayscale (or color distance)
    const gray = this.toGrayscale(srcData, width, height, paperColor);
    
    // Step 2: Gaussian blur for noise reduction
    const blurred = this.gaussianBlur(gray, width, height);
    
    // Step 3: Sobel gradients
    const { magnitude, direction, maxMag } = this.sobelGradients(blurred, width, height);
    
    // Step 4: Non-maximum suppression
    const thinned = this.nonMaxSuppression(magnitude, direction, width, height);
    
    // Step 5: Double threshold + hysteresis
    const edges = this.hysteresisThreshold(thinned, width, height, lowRatio, highRatio);
    
    return { edges, magnitude, maxMag };
  }

  // Trace edge contour in order using boundary following (Moore neighborhood)
  static traceEdgeContour(edges, width, height) {
    const visited = new Uint8Array(width * height);
    const contours = [];
    // 8-connectivity directions: clockwise from right
    const DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
    
    // Find all starting points and trace ordered contours
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        if (edges[idx] === 1 && visited[idx] === 0) {
          // Moore boundary tracing - follows contour in order
          const contour = [];
          let cx = x, cy = y;
          let dir = 0; // Start looking right
          const startX = x, startY = y;
          let steps = 0;
          const maxSteps = width * height;
          
          do {
            contour.push({ x: cx, y: cy });
            visited[cy * width + cx] = 1;
            
            // Look for next edge pixel, starting from direction opposite to where we came from
            let found = false;
            const startDir = (dir + 5) % 8; // Start search from dir+5 (backtrack + 1 CCW)
            
            for (let i = 0; i < 8; i++) {
              const checkDir = (startDir + i) % 8;
              const nx = cx + DIRS[checkDir][0];
              const ny = cy + DIRS[checkDir][1];
              
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (edges[ny * width + nx] === 1) {
                  cx = nx;
                  cy = ny;
                  dir = checkDir;
                  found = true;
                  break;
                }
              }
            }
            
            if (!found) break;
            steps++;
          } while ((cx !== startX || cy !== startY) && steps < maxSteps);
          
          if (contour.length > 50) {
            contours.push(contour);
          }
        }
      }
    }
    
    // Return largest contour
    contours.sort((a, b) => b.length - a.length);
    return contours.length > 0 ? contours[0] : [];
  }

  // Simplify contour using Douglas-Peucker algorithm
  static simplifyContour(points, epsilon) {
    if (points.length < 3) return points;
    
    const getSqDist = (p, p1, p2) => {
      let x = p1.x, y = p1.y;
      let dx = p2.x - x, dy = p2.y - y;
      if (dx !== 0 || dy !== 0) {
        const t = ((p.x - x) * dx + (p.y - y) * dy) / (dx * dx + dy * dy);
        if (t > 1) { x = p2.x; y = p2.y; }
        else if (t > 0) { x += dx * t; y += dy * t; }
      }
      dx = p.x - x; dy = p.y - y;
      return dx * dx + dy * dy;
    };
    
    const simplifyDP = (pts, first, last, sqTol, simplified) => {
      let maxSqDist = 0, index = 0;
      for (let i = first + 1; i < last; i++) {
        const sqDist = getSqDist(pts[i], pts[first], pts[last]);
        if (sqDist > maxSqDist) { maxSqDist = sqDist; index = i; }
      }
      if (maxSqDist > sqTol) {
        if (index - first > 1) simplifyDP(pts, first, index, sqTol, simplified);
        simplified.push(pts[index]);
        if (last - index > 1) simplifyDP(pts, index, last, sqTol, simplified);
      }
    };
    
    const sqTol = epsilon * epsilon;
    const simplified = [points[0]];
    simplifyDP(points, 0, points.length - 1, sqTol, simplified);
    simplified.push(points[points.length - 1]);
    
    return simplified;
  }

  // Find 4 corners from a contour using corner detection
  static findQuadrilateralCorners(contour, width, height) {
    if (contour.length < 4) {
      console.log('Contour too short:', contour.length);
      return null;
    }
    
    // Method 1: Convex hull + extremes approach
    // Find the 4 extreme points (TL, TR, BR, BL)
    let minSum = Infinity, maxSum = -Infinity;
    let minDiff = Infinity, maxDiff = -Infinity;
    let tl = null, tr = null, br = null, bl = null;
    
    for (const p of contour) {
      const sum = p.x + p.y;
      const diff = p.x - p.y;
      
      if (sum < minSum) { minSum = sum; tl = { ...p }; }
      if (sum > maxSum) { maxSum = sum; br = { ...p }; }
      if (diff > maxDiff) { maxDiff = diff; tr = { ...p }; }
      if (diff < minDiff) { minDiff = diff; bl = { ...p }; }
    }
    
    if (!tl || !tr || !br || !bl) {
      console.log('Missing corners:', { tl, tr, br, bl });
      return null;
    }
    
    // Validate: corners should form a reasonable quadrilateral
    const corners = [tl, tr, br, bl];
    
    // Check minimum area (at least 1% of image area - reduced for flexibility)
    const area = this.polygonArea(corners);
    const minArea = width * height * 0.01;
    console.log('Corner validation - area:', area, 'minArea:', minArea);
    if (area < minArea) {
      console.log('Area too small');
      return null;
    }
    
    // Check that corners are reasonably spread apart
    const distances = [];
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      const dx = corners[j].x - corners[i].x;
      const dy = corners[j].y - corners[i].y;
      distances.push(Math.sqrt(dx * dx + dy * dy));
    }
    const minDist = Math.min(...distances);
    const maxDist = Math.max(...distances);
    console.log('Edge distances:', distances, 'ratio:', maxDist / minDist);
    
    // Sides should be roughly similar in length (ratio < 5:1)
    if (maxDist / minDist > 5) {
      console.log('Edge ratio too large');
      return null;
    }
    
    // Check convexity
    if (!this.isConvex(corners)) {
      console.log('Not convex');
      return null;
    }
    
    console.log('Valid corners found:', corners);
    return corners;
  }

  // Calculate polygon area using shoelace formula
  static polygonArea(corners) {
    let area = 0;
    const n = corners.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += corners[i].x * corners[j].y;
      area -= corners[j].x * corners[i].y;
    }
    return Math.abs(area) / 2;
  }

  // Check if polygon is convex
  static isConvex(corners) {
    const n = corners.length;
    let sign = 0;
    for (let i = 0; i < n; i++) {
      const p1 = corners[i];
      const p2 = corners[(i + 1) % n];
      const p3 = corners[(i + 2) % n];
      const cross = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x);
      if (cross !== 0) {
        if (sign === 0) sign = cross > 0 ? 1 : -1;
        else if ((cross > 0 ? 1 : -1) !== sign) return false;
      }
    }
    return true;
  }

  // Main function: detect paper corners from image
  static detectPaperCorners(srcData, width, height, paperColor = null) {
    // Try multiple threshold settings for robustness
    const thresholdConfigs = [
      { low: 0.03, high: 0.10 },  // More sensitive
      { low: 0.02, high: 0.08 },  // Even more sensitive
      { low: 0.05, high: 0.15 },  // Standard
    ];
    
    for (const config of thresholdConfigs) {
      // Run Canny edge detection with current thresholds
      const { edges } = this.cannyEdgeDetection(srcData, width, height, paperColor, config.low, config.high);
      
      // Count edge pixels for debugging
      let edgeCount = 0;
      for (let i = 0; i < edges.length; i++) {
        if (edges[i] === 1) edgeCount++;
      }
      console.log(`Edge detection with thresholds (${config.low}, ${config.high}): ${edgeCount} edge pixels`);
      
      // Trace the largest edge contour
      const contour = this.traceEdgeContour(edges, width, height);
      console.log(`Contour length: ${contour.length}`);
      
      if (contour.length < 30) continue; // Try next threshold config
      
      // Simplify the contour
      const simplified = this.simplifyContour(contour, 3);
      console.log(`Simplified contour length: ${simplified.length}`);
      
      // Find quadrilateral corners
      const corners = this.findQuadrilateralCorners(simplified, width, height);
      
      if (corners) {
        console.log('Found valid corners with config:', config);
        return corners;
      }
    }
    
    console.log('Edge detection failed with all threshold configs');
    return null;
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
            <p className="absolute bottom-6 text-neutral-600 text-[10px] tracking-wider">Hotwave studio</p>
        </div>
    );
};

/**
 * TUTORIAL STEPS DATA
 */
const TUTORIAL_STEPS = [
    {
        title: "Welcome to ShapeScanner",
        description: "This app converts photos of physical objects into vector files (DXF/SVG) for CAD and CNC manufacturing. Let's walk through the key features!",
        icon: "welcome"
    },
    {
        title: "Step 1: Capture",
        description: "Start by taking a photo or uploading an image. Place your object on a sheet of paper with good contrast. The paper edges help with perspective correction.",
        icon: "camera"
    },
    {
        title: "Step 2: Calibrate",
        description: "Drag the corner markers to align with the paper edges. Adjust contrast if needed, or use the paper color picker for tricky backgrounds. Set paper size (A4, Letter, etc.).",
        icon: "calibrate"
    },
    {
        title: "Step 3: Process",
        description: "The app detects shapes automatically. Use view modes to see the original, processed outline, or detection heatmap. Multiple shapes are detected and numbered.",
        icon: "process"
    },
    {
        title: "Detection Settings",
        description: "• Threshold: Sensitivity for detecting object edges\n• Shadow Removal: Removes dark shadows around objects\n• Noise Filter: Smooths out small artifacts\n• Detail Scan: Resolution of edge detection",
        icon: "settings"
    },
    {
        title: "Per-Shape Settings",
        description: "Tap on any shape to select it. Open 'Advanced' settings to fine-tune each shape independently. Each shape can have its own threshold, smoothing, and refinement settings.",
        icon: "shapes"
    },
    {
        title: "Export Options",
        description: "• DXF: For CAD software (AutoCAD, Fusion 360)\n• SVG: For graphic design and laser cutters\n• PNG: Simple image export\n\nAll vector exports use real millimeter dimensions!",
        icon: "export"
    }
];

/**
 * TUTORIAL OVERLAY COMPONENT
 */
const TutorialOverlay = ({ currentStep, totalSteps, stepData, onNext, onPrev, onClose, onDontShowAgain }) => {
    const getIcon = (iconType) => {
        switch(iconType) {
            case 'welcome': return <BookOpen size={32} className="text-[var(--accent-blue)]"/>;
            case 'camera': return <CameraIcon size={32} className="text-[var(--accent-blue)]"/>;
            case 'calibrate': return <Maximize2 size={32} className="text-[var(--accent-blue)]"/>;
            case 'process': return <ScanLine size={32} className="text-[var(--accent-blue)]"/>;
            case 'settings': return <Settings size={32} className="text-[var(--accent-blue)]"/>;
            case 'shapes': return <Layers size={32} className="text-[var(--accent-emerald)]"/>;
            case 'export': return <Download size={32} className="text-[var(--accent-emerald)]"/>;
            default: return <HelpCircle size={32} className="text-[var(--accent-blue)]"/>;
        }
    };

    return (
        <div className="fixed inset-0 bg-[var(--bg-overlay)] backdrop-blur-sm flex items-center justify-center z-[90] p-4">
            <div className="theme-bg-secondary border theme-border rounded-2xl max-w-md w-full shadow-2xl overflow-hidden">
                <div className="p-6">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="w-14 h-14 rounded-xl theme-bg-tertiary flex items-center justify-center">
                            {getIcon(stepData.icon)}
                        </div>
                        <div className="flex-1">
                            <h2 className="text-lg font-bold theme-text-primary">{stepData.title}</h2>
                            <p className="text-xs theme-text-muted">Step {currentStep + 1} of {totalSteps}</p>
                        </div>
                        <button onClick={onClose} className="p-2 hover:theme-bg-tertiary rounded-lg transition-colors">
                            <X size={20} className="theme-text-secondary"/>
                        </button>
                    </div>
                    
                    <p className="text-sm theme-text-secondary leading-relaxed whitespace-pre-line mb-6">
                        {stepData.description}
                    </p>
                    
                    <div className="flex items-center gap-2 mb-4">
                        {Array.from({ length: totalSteps }).map((_, i) => (
                            <div 
                                key={i} 
                                className={`h-1.5 flex-1 rounded-full transition-colors ${i === currentStep ? 'bg-[var(--accent-blue)]' : i < currentStep ? 'bg-[var(--accent-blue)]/50' : 'theme-bg-tertiary'}`}
                            />
                        ))}
                    </div>
                </div>
                
                <div className="theme-bg-tertiary px-6 py-4 flex items-center justify-between">
                    <button 
                        onClick={onDontShowAgain}
                        className="text-xs theme-text-muted hover:theme-text-secondary transition-colors"
                    >
                        Don't show again
                    </button>
                    
                    <div className="flex gap-2">
                        {currentStep > 0 && (
                            <button 
                                onClick={onPrev}
                                className="px-4 py-2 rounded-lg theme-bg-secondary hover:opacity-80 theme-text-primary text-sm font-medium flex items-center gap-1 transition-colors"
                            >
                                <ChevronLeft size={16}/> Back
                            </button>
                        )}
                        <button 
                            onClick={currentStep === totalSteps - 1 ? onClose : onNext}
                            className="px-4 py-2 rounded-lg bg-[var(--accent-blue)] hover:bg-[var(--accent-blue-hover)] text-white text-sm font-medium flex items-center gap-1 transition-colors"
                        >
                            {currentStep === totalSteps - 1 ? 'Get Started' : <>Next <ChevronRight size={16}/></>}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

/**
 * SETTINGS MENU COMPONENT
 */
const SettingsMenu = ({ isOpen, onClose, showTutorialOnStart, setShowTutorialOnStart, onOpenTutorial, isDarkTheme, setIsDarkTheme }) => {
    if (!isOpen) return null;
    
    return (
        <div className="fixed inset-0 bg-[var(--bg-overlay)] backdrop-blur-sm flex items-center justify-center z-[80] p-4" onClick={onClose}>
            <div className="theme-bg-card border theme-border rounded-2xl max-w-sm w-full shadow-2xl" onClick={e => e.stopPropagation()}>
                <div className="p-4 border-b theme-border-secondary flex items-center justify-between">
                    <h2 className="text-lg font-bold theme-text-primary flex items-center gap-2">
                        <Settings size={20} className="text-[var(--accent-blue)]"/> Settings
                    </h2>
                    <button onClick={onClose} className="p-2 hover:theme-bg-tertiary rounded-lg transition-colors">
                        <X size={20} className="theme-text-secondary"/>
                    </button>
                </div>
                
                <div className="p-4 space-y-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium theme-text-primary">Dark Theme</p>
                            <p className="text-xs theme-text-muted">Switch between dark and light mode</p>
                        </div>
                        <button 
                            onClick={() => setIsDarkTheme(!isDarkTheme)}
                            className={`w-12 h-7 rounded-full transition-colors relative ${isDarkTheme ? 'bg-[var(--accent-blue)]' : 'bg-neutral-300'}`}
                        >
                            <div className={`absolute top-1 w-5 h-5 rounded-full bg-white shadow transition-transform ${isDarkTheme ? 'translate-x-6' : 'translate-x-1'}`}/>
                        </button>
                    </div>
                    
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium theme-text-primary">Show Tutorial on Start</p>
                            <p className="text-xs theme-text-muted">Display walkthrough for new users</p>
                        </div>
                        <button 
                            onClick={() => setShowTutorialOnStart(!showTutorialOnStart)}
                            className={`w-12 h-7 rounded-full transition-colors relative ${showTutorialOnStart ? 'bg-[var(--accent-blue)]' : 'bg-[var(--bg-tertiary)]'}`}
                        >
                            <div className={`absolute top-1 w-5 h-5 rounded-full bg-white shadow transition-transform ${showTutorialOnStart ? 'translate-x-6' : 'translate-x-1'}`}/>
                        </button>
                    </div>
                    
                    <button 
                        onClick={() => { onOpenTutorial(); onClose(); }}
                        className="w-full py-3 rounded-xl theme-bg-tertiary hover:opacity-80 border theme-border theme-text-primary text-sm font-medium flex items-center justify-center gap-2 transition-colors"
                    >
                        <BookOpen size={18}/> View Tutorial
                    </button>
                </div>
                
                <div className="p-4 border-t theme-border-secondary">
                    <p className="text-[10px] theme-text-muted text-center">ShapeScanner v1.0 • Hotwave studio</p>
                </div>
            </div>
        </div>
    );
};

/**
 * MAIN COMPONENT
 */
const ShapeScanner = () => {
  // --- SPLASH STATE ---
  const [showSplash, setShowSplash] = useState(true);

  // --- TUTORIAL STATE ---
  const [showTutorial, setShowTutorial] = useState(false);
  const [tutorialStep, setTutorialStep] = useState(0);
  const [showTutorialOnStart, setShowTutorialOnStart] = useState(() => {
    const stored = localStorage.getItem('shapescanner_show_tutorial');
    return stored === null ? true : stored === 'true';
  });
  const [showSettingsMenu, setShowSettingsMenu] = useState(false);
  
  // --- THEME STATE ---
  const [isDarkTheme, setIsDarkTheme] = useState(() => {
    const stored = localStorage.getItem('shapescanner_dark_theme');
    return stored === null ? true : stored === 'true';
  });

  // Save theme preference to localStorage
  useEffect(() => {
    localStorage.setItem('shapescanner_dark_theme', isDarkTheme.toString());
  }, [isDarkTheme]);

  // Check if first time user and show tutorial after splash
  useEffect(() => {
    if (!showSplash && showTutorialOnStart) {
      const hasSeenTutorial = localStorage.getItem('shapescanner_tutorial_seen');
      if (!hasSeenTutorial) {
        setShowTutorial(true);
      }
    }
  }, [showSplash, showTutorialOnStart]);

  // Save tutorial preference to localStorage
  useEffect(() => {
    localStorage.setItem('shapescanner_show_tutorial', showTutorialOnStart.toString());
  }, [showTutorialOnStart]);

  const handleTutorialClose = () => {
    setShowTutorial(false);
    setTutorialStep(0);
    localStorage.setItem('shapescanner_tutorial_seen', 'true');
  };

  const handleDontShowAgain = () => {
    setShowTutorialOnStart(false);
    handleTutorialClose();
  };

  const openTutorial = () => {
    setTutorialStep(0);
    setShowTutorial(true);
  };

  // --- APP STATE ---
  const [step, setStep] = useState('capture');
  const [imageSrc, setImageSrc] = useState(null);
  const [imgDims, setImgDims] = useState({ w: 0, h: 0 });
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Debug State
  const [showDebug, setShowDebug] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [debugStats, setDebugStats] = useState({ mappedPixels: 0, matrixValid: false, processingTime: 0 });

  // Calibration
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [calMonochrome, setCalMonochrome] = useState(false);
  const [calContrast, setCalContrast] = useState(100);
  const [corners, setCorners] = useState([]); 
  const [activeCorner, setActiveCorner] = useState(null);
  const [paperColor, setPaperColor] = useState({ r: 255, g: 255, b: 255 }); // Default white paper
  const [isPickingPaperColor, setIsPickingPaperColor] = useState(false);
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
  
  // Multi-polygon state with per-polygon settings
  const [detectedPolygons, setDetectedPolygons] = useState([]);
  const [selectedPolygonIndex, setSelectedPolygonIndex] = useState(0);
  
  // Settings scope: 'global' applies to all polygons, 'polygon' applies to selected only
  const [settingsScope, setSettingsScope] = useState('global');
  
  // Default/global settings that new polygons inherit
  const defaultSettings = {
    threshold: 30,
    scanStep: 2,
    curveSmoothing: 2,
    noiseFilter: 2,
    shadowRemoval: 0,
    smartRefine: true,
    invertResult: false,
    showHoles: true
  };
  
  // Helper to get current polygon's settings or global defaults
  const getSelectedPolygonSettings = useCallback(() => {
    if (detectedPolygons.length > 0 && detectedPolygons[selectedPolygonIndex]?.settings) {
      return detectedPolygons[selectedPolygonIndex].settings;
    }
    return { threshold, scanStep, curveSmoothing, noiseFilter, shadowRemoval, smartRefine, invertResult, showHoles: true };
  }, [detectedPolygons, selectedPolygonIndex, threshold, scanStep, curveSmoothing, noiseFilter, shadowRemoval, smartRefine, invertResult]);
  
  // Get current settings based on scope
  const getCurrentSettings = useCallback(() => {
    if (settingsScope === 'polygon' && detectedPolygons.length > 0 && detectedPolygons[selectedPolygonIndex]?.settings) {
      return detectedPolygons[selectedPolygonIndex].settings;
    }
    return { threshold, scanStep, curveSmoothing, noiseFilter, shadowRemoval, smartRefine, invertResult, showHoles: true };
  }, [settingsScope, detectedPolygons, selectedPolygonIndex, threshold, scanStep, curveSmoothing, noiseFilter, shadowRemoval, smartRefine, invertResult]);
  
  // Update settings based on current scope
  const updateSetting = useCallback((key, value) => {
    // Detection settings require reprocessPolygonDetection, vector settings require reprocessPolygon
    const detectionSettingKeys = ['threshold', 'shadowRemoval', 'noiseFilter', 'scanStep'];
    const isDetectionSetting = detectionSettingKeys.includes(key);
    
    if (settingsScope === 'polygon' && detectedPolygons.length > 0) {
      // Per-polygon mode: update only selected polygon
      setDetectedPolygons(prev => {
        const updated = [...prev];
        if (updated[selectedPolygonIndex]) {
          updated[selectedPolygonIndex] = {
            ...updated[selectedPolygonIndex],
            settings: {
              ...updated[selectedPolygonIndex].settings,
              [key]: value
            },
            ...(isDetectionSetting ? { needsDetectionReprocess: true } : { needsReprocess: true })
          };
        }
        return updated;
      });
    } else {
      // Global mode: update global state (triggers full reprocessing)
      switch(key) {
        case 'threshold': setThreshold(value); break;
        case 'scanStep': setScanStep(value); break;
        case 'curveSmoothing': setCurveSmoothing(value); break;
        case 'noiseFilter': setNoiseFilter(value); break;
        case 'shadowRemoval': setShadowRemoval(value); break;
        case 'smartRefine': setSmartRefine(value); break;
        case 'invertResult': setInvertResult(value); break;
      }
    }
  }, [settingsScope, detectedPolygons.length, selectedPolygonIndex]);
  
  // Apply current global settings to all polygons
  const applyGlobalToAll = useCallback(() => {
    if (detectedPolygons.length === 0) return;
    const globalSettings = { threshold, scanStep, curveSmoothing, noiseFilter, shadowRemoval, smartRefine, invertResult, showHoles: true };
    setDetectedPolygons(prev => prev.map(poly => ({
      ...poly,
      settings: { ...globalSettings },
      needsDetectionReprocess: true
    })));
  }, [detectedPolygons.length, threshold, scanStep, curveSmoothing, noiseFilter, shadowRemoval, smartRefine, invertResult]);
  
  // Reset selected polygon to global defaults
  const resetPolygonToDefaults = useCallback(() => {
    if (detectedPolygons.length === 0) return;
    const globalSettings = { threshold, scanStep, curveSmoothing, noiseFilter, shadowRemoval, smartRefine, invertResult, showHoles: true };
    setDetectedPolygons(prev => {
      const updated = [...prev];
      if (updated[selectedPolygonIndex]) {
        updated[selectedPolygonIndex] = {
          ...updated[selectedPolygonIndex],
          settings: { ...globalSettings },
          needsDetectionReprocess: true
        };
      }
      return updated;
    });
  }, [detectedPolygons.length, selectedPolygonIndex, threshold, scanStep, curveSmoothing, noiseFilter, shadowRemoval, smartRefine, invertResult]);
  
  // Process view zoom/pan state
  const [processView, setProcessView] = useState({ x: 0, y: 0, scale: 1 });
  const [isProcessPanning, setIsProcessPanning] = useState(false);
  const [lastProcessPos, setLastProcessPos] = useState({ x: 0, y: 0 });
  
  // Draggable badge state
  const [badgePosition, setBadgePosition] = useState({ x: null, y: 16 }); // null x means right-aligned
  const [isDraggingBadge, setIsDraggingBadge] = useState(false);
  const [badgeDragOffset, setBadgeDragOffset] = useState({ x: 0, y: 0 });
  
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
  
  // Save Dialog
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveFileName, setSaveFileName] = useState('');
  const [saveType, setSaveType] = useState('dxf');

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
          setDetectedPolygons([]); setSelectedPolygonIndex(0);
          
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

  // Camera capture handler for native platforms
  const handleCameraCapture = async () => {
    try {
      const photo = await Camera.getPhoto({
        quality: 90,
        allowEditing: false,
        resultType: CameraResultType.DataUrl,
        source: CameraSource.Camera
      });
      
      if (photo.dataUrl) {
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
          setImageSrc(photo.dataUrl);
          
          // Reset
          setCalMonochrome(false); setCalContrast(100);
          setSegmentMode('auto'); setInvertResult(false);
          setProcessedPath([]); setObjectDims(null);
          setViewMode('original');
          setAiResult(null); setShowAiPanel(false);
          setDetectedPolygons([]); setSelectedPolygonIndex(0);
          
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = w; tempCanvas.height = h;
          const ctx = tempCanvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);
          sourcePixelData.current = ctx.getImageData(0, 0, w, h).data;
          
          setStep('calibrate');
        };
        img.src = photo.dataUrl;
      }
    } catch (error) {
      console.error('Camera error:', error);
      // Fallback to file input with capture attribute
      if (fileInputRef.current) {
        fileInputRef.current.setAttribute('capture', 'environment');
        fileInputRef.current.click();
        fileInputRef.current.removeAttribute('capture');
      }
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

  const zoomToCorner = useCallback((cornerIndex) => {
    if (!canvasRef.current || cornerIndex === null) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const corner = corners[cornerIndex];
    const zoomScale = 3;
    const newX = rect.width / 2 - corner.x * zoomScale;
    const newY = rect.height / 2 - corner.y * zoomScale;
    setView({ x: newX, y: newY, scale: zoomScale });
  }, [corners]);

  const handleStart = (clientX, clientY) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = clientX - rect.left; const sy = clientY - rect.top;
    const p = toImageCoords(sx, sy);
    
    // Handle paper color picking
    if (isPickingPaperColor && sourcePixelData.current) {
      const ix = Math.floor(p.x); const iy = Math.floor(p.y);
      if (ix >= 0 && ix < imgDims.w && iy >= 0 && iy < imgDims.h) {
        const i = (iy * imgDims.w + ix) * 4;
        setPaperColor({ r: sourcePixelData.current[i], g: sourcePixelData.current[i+1], b: sourcePixelData.current[i+2] });
        setIsPickingPaperColor(false);
      }
      return;
    }
    
    const hitRadius = 30 / view.scale; 
    let closest = -1; let minD = Infinity;
    corners.forEach((c, i) => {
      const d = Math.sqrt((c.x - p.x)**2 + (c.y - p.y)**2);
      if (d < hitRadius && d < minD) { minD = d; closest = i; }
    });
    if (closest !== -1) {
      setActiveCorner(closest);
      zoomToCorner(closest);
    }
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

  const handleEnd = () => { 
    if (activeCorner !== null) {
      fitToScreen();
    }
    setActiveCorner(null); 
    setIsPanning(false); 
  };
  
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

  // --- AUTO DETECT (uses edge detection with Canny + contour fitting) ---
  const autoDetectCorners = useCallback(() => {
    if (!sourcePixelData.current) return;
    const width = imgDims.w; const height = imgDims.h; const srcData = sourcePixelData.current;
    
    // Apply contrast adjustment to match what user sees
    const contrastFactor = calContrast / 100;
    const applyContrast = (value) => {
      const adjusted = ((value / 255 - 0.5) * contrastFactor + 0.5) * 255;
      return Math.max(0, Math.min(255, adjusted));
    };
    
    // Create contrast-adjusted copy of the image data for detection
    const adjustedData = new Uint8ClampedArray(srcData.length);
    for (let i = 0; i < srcData.length; i += 4) {
      adjustedData[i] = applyContrast(srcData[i]);
      adjustedData[i + 1] = applyContrast(srcData[i + 1]);
      adjustedData[i + 2] = applyContrast(srcData[i + 2]);
      adjustedData[i + 3] = srcData[i + 3];
    }
    
    console.log('Auto-detect running with contrast:', calContrast, 'paper color:', paperColor);
    
    // Try edge-based detection first (more accurate for paper boundaries)
    const edgeCorners = EdgeDetector.detectPaperCorners(adjustedData, width, height, paperColor);
    
    console.log('Edge detection result:', edgeCorners);
    
    if (edgeCorners && edgeCorners.length === 4) {
      // Validate the detected corners
      const area = EdgeDetector.polygonArea(edgeCorners);
      const minArea = width * height * 0.05;
      
      console.log('Corner validation - area:', area, 'minArea:', minArea, 'isConvex:', EdgeDetector.isConvex(edgeCorners));
      
      if (area >= minArea && EdgeDetector.isConvex(edgeCorners)) {
        console.log('Using edge-detected corners');
        setCorners(edgeCorners);
        return;
      }
    }
    
    console.log('Falling back to color-distance detection');
    
    // Fallback to color-distance based detection if edge detection fails
    const getColorDistance = (r, g, b) => {
      return Math.sqrt((r - paperColor.r)**2 + (g - paperColor.g)**2 + (b - paperColor.b)**2);
    };
    
    // Build histogram of color distances to find optimal threshold
    const maxDist = 442; // sqrt(255^2 * 3)
    const histogram = new Array(256).fill(0);
    const step = 4;
    for(let i=0; i<adjustedData.length; i+=4*step) {
        const dist = getColorDistance(adjustedData[i], adjustedData[i+1], adjustedData[i+2]);
        const normalized = Math.round((dist / maxDist) * 255);
        histogram[Math.max(0, Math.min(255, normalized))]++;
    }

    // Otsu's method on color distance histogram
    let sum = 0; for (let i = 0; i < 256; i++) sum += i * histogram[i];
    let sumB = 0; let wB = 0; let wF = 0; let maxVar = 0; let otsuThreshold = 0;
    const total = (adjustedData.length / 4) / step;

    for (let i = 0; i < 256; i++) {
        wB += histogram[i]; if (wB === 0) continue;
        wF = total - wB; if (wF === 0) break;
        sumB += i * histogram[i];
        const mB = sumB / wB; const mF = (sum - sumB) / wF;
        const varBetween = wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) { maxVar = varBetween; otsuThreshold = i; }
    }
    
    // Convert back to actual distance threshold
    const distThreshold = (otsuThreshold / 255) * maxDist;
    console.log('Otsu threshold:', otsuThreshold, 'distThreshold:', distThreshold);

    let tl = { val: Infinity, x: 0, y: 0 }; let tr = { val: -Infinity, x: 0, y: 0 };
    let br = { val: -Infinity, x: 0, y: 0 }; let bl = { val: Infinity, x: 0, y: 0 };
    const padding = Math.min(width, height) * 0.05; 

    for (let y = padding; y < height - padding; y += step) {
      for (let x = padding; x < width - padding; x += step) {
        const i = (Math.floor(y) * width + Math.floor(x)) * 4;
        const dist = getColorDistance(adjustedData[i], adjustedData[i+1], adjustedData[i+2]);
        
        // Pixel is "paper" if its color is close to the selected paper color
        if (dist < distThreshold) { 
          const sum = x + y; const diff = x - y;
          if (sum < tl.val) { tl.val = sum; tl.x = x; tl.y = y; }
          if (diff > tr.val) { tr.val = diff; tr.x = x; tr.y = y; }
          if (sum > br.val) { br.val = sum; br.x = x; br.y = y; }
          if (diff < bl.val) { bl.val = diff; bl.x = x; bl.y = y; }
        }
      }
    }
    console.log('Fallback corners found:', tl.val !== Infinity ? 'yes' : 'no');
    if (tl.val !== Infinity) { setCorners([{ x: tl.x, y: tl.y }, { x: tr.x, y: tr.y }, { x: br.x, y: br.y }, { x: bl.x, y: bl.y }]); }
  }, [imgDims, paperColor, calContrast]);


  // --- POINT IN POLYGON TEST ---
  const pointInPolygon = useCallback((point, polygon) => {
    if (!polygon || polygon.length < 3) return false;
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x, yi = polygon[i].y;
      const xj = polygon[j].x, yj = polygon[j].y;
      if (((yi > point.y) !== (yj > point.y)) && (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }, []);

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
    
    // Check if tapped on a polygon (tap-to-select)
    if (detectedPolygons.length > 0 && unwarpedBufferRef.current) {
      const { width: bufWidth, height: bufHeight } = unwarpedBufferRef.current;
      // Convert pixel coords to mm coords for hit testing
      const mmX = (px / bufWidth) * paperWidth;
      const mmY = ((bufHeight - py) / bufHeight) * paperHeight;
      const clickPoint = { x: mmX, y: mmY };
      
      // Check polygons in reverse order (so topmost/smallest is selected first)
      for (let i = detectedPolygons.length - 1; i >= 0; i--) {
        const poly = detectedPolygons[i];
        if (pointInPolygon(clickPoint, poly.outer)) {
          setSelectedPolygonIndex(i);
          setProcessedPath(poly.outer);
          setDetectedShapeType(poly.type);
          const minX = Math.min(...poly.outer.map(p => p.x));
          const maxX = Math.max(...poly.outer.map(p => p.x));
          const minY = Math.min(...poly.outer.map(p => p.y));
          const maxY = Math.max(...poly.outer.map(p => p.y));
          setObjectDims({ width: maxX - minX, height: maxY - minY, minX, maxX, minY, maxY });
          return; // Don't start drag selection if we selected a polygon
        }
      }
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

  // --- PROCESS VIEW ZOOM/PAN HANDLERS ---
  const handleProcessViewWheel = (e) => {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setProcessView(v => ({
      scale: Math.max(0.5, Math.min(5, v.scale * delta)),
      x: x - (x - v.x) * delta,
      y: y - (y - v.y) * delta
    }));
  };

  const handleProcessPanStart = (e) => {
    setIsProcessPanning(true);
    setLastProcessPos({ x: e.clientX, y: e.clientY });
  };

  const handleProcessPanMove = (e) => {
    if (!isProcessPanning) return;
    setProcessView(v => ({
      ...v,
      x: v.x + e.clientX - lastProcessPos.x,
      y: v.y + e.clientY - lastProcessPos.y
    }));
    setLastProcessPos({ x: e.clientX, y: e.clientY });
  };

  const handleProcessPanEnd = () => {
    setIsProcessPanning(false);
  };

  const resetProcessView = () => {
    setProcessView({ x: 0, y: 0, scale: 1 });
  };

  // --- DRAGGABLE BADGE HANDLERS ---
  const handleBadgeDragStart = (e) => {
    e.stopPropagation();
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    setBadgeDragOffset({ 
      x: clientX - rect.left, 
      y: clientY - rect.top 
    });
    setIsDraggingBadge(true);
  };

  const handleBadgeDragMove = useCallback((e) => {
    if (!isDraggingBadge) return;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    
    // Constrain to viewport bounds with padding
    const padding = 16;
    const badgeWidth = 150; // Approximate width
    const badgeHeight = 40; // Approximate height
    const maxX = window.innerWidth - badgeWidth - padding;
    const maxY = window.innerHeight - badgeHeight - padding;
    
    const newX = Math.max(padding, Math.min(maxX, clientX - badgeDragOffset.x));
    const newY = Math.max(padding, Math.min(maxY, clientY - badgeDragOffset.y));
    
    setBadgePosition({ x: newX, y: newY });
  }, [isDraggingBadge, badgeDragOffset]);

  const resetBadgePosition = useCallback(() => {
    setBadgePosition({ x: null, y: 16 }); // Reset to default right-aligned position
  }, []);

  const handleBadgeDragEnd = useCallback(() => {
    setIsDraggingBadge(false);
  }, []);

  // Global mouse/touch move and up listeners for badge dragging
  useEffect(() => {
    if (isDraggingBadge) {
      const handleMove = (e) => handleBadgeDragMove(e);
      const handleEnd = () => handleBadgeDragEnd();
      window.addEventListener('mousemove', handleMove);
      window.addEventListener('mouseup', handleEnd);
      window.addEventListener('touchmove', handleMove);
      window.addEventListener('touchend', handleEnd);
      return () => {
        window.removeEventListener('mousemove', handleMove);
        window.removeEventListener('mouseup', handleEnd);
        window.removeEventListener('touchmove', handleMove);
        window.removeEventListener('touchend', handleEnd);
      };
    }
  }, [isDraggingBadge, handleBadgeDragMove, handleBadgeDragEnd]);

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

    // Multi-polygon detection with holes
    if (viewMode !== 'original') {
        // Use ContourTracer to detect all polygons with holes
        const polygons = ContourTracer.detectPolygons(mask, targetW, targetH, paperWidth, paperHeight, scanStep);
        
        // Process and refine each polygon
        const refinedPolygons = polygons.map(poly => {
          let outerPoints = poly.outer;
          let detected = 'poly';
          
          if (smartRefine && outerPoints.length > 0) {
            const circleFit = ShapeFitter.fitCircle(outerPoints);
            if (circleFit) {
              outerPoints = ShapeFitter.generateCircle(circleFit.cx, circleFit.cy, circleFit.r);
              detected = 'circle';
              poly.bbox.minX = circleFit.cx - circleFit.r;
              poly.bbox.maxX = circleFit.cx + circleFit.r;
              poly.bbox.minY = circleFit.cy - circleFit.r;
              poly.bbox.maxY = circleFit.cy + circleFit.r;
            } else {
              outerPoints = VectorUtils.simplify(outerPoints, 0.5);
              outerPoints = VectorUtils.smooth(outerPoints, curveSmoothing);
            }
          }
          
          // Refine holes as well
          const refinedHoles = poly.holes.map(hole => {
            let holePoints = hole;
            if (smartRefine) {
              const holeFit = ShapeFitter.fitCircle(hole);
              if (holeFit) {
                holePoints = ShapeFitter.generateCircle(holeFit.cx, holeFit.cy, holeFit.r);
              } else {
                holePoints = VectorUtils.simplify(holePoints, 0.5);
                holePoints = VectorUtils.smooth(holePoints, curveSmoothing);
              }
            }
            return holePoints;
          });
          
          // Calculate pixel-space bounding box for per-polygon reprocessing
          const pixelBbox = {
            minX: Math.floor((poly.bbox.minX / paperWidth) * targetW),
            maxX: Math.ceil((poly.bbox.maxX / paperWidth) * targetW),
            minY: Math.floor((1 - poly.bbox.maxY / paperHeight) * targetH),
            maxY: Math.ceil((1 - poly.bbox.minY / paperHeight) * targetH)
          };
          
          return {
            id: Date.now() + Math.random(),
            outer: outerPoints,
            holes: refinedHoles,
            bbox: poly.bbox,
            pixelBbox,
            type: detected,
            rawOuter: poly.outer,
            rawHoles: poly.holes,
            settings: {
              threshold,
              scanStep,
              curveSmoothing,
              noiseFilter,
              shadowRemoval,
              smartRefine,
              invertResult,
              showHoles: true
            },
            needsReprocess: false
          };
        });
        
        // Preserve settings from previous polygons by matching bounding boxes
        setDetectedPolygons(prev => {
          if (prev.length === 0) return refinedPolygons;
          
          return refinedPolygons.map((newPoly, idx) => {
            // Try to find a matching polygon from previous state
            const matchingPrev = prev.find((oldPoly, oldIdx) => {
              // Match by index if same count, or by overlapping bounding box
              if (prev.length === refinedPolygons.length && oldIdx === idx) return true;
              const overlap = !(newPoly.bbox.maxX < oldPoly.bbox.minX || 
                               newPoly.bbox.minX > oldPoly.bbox.maxX ||
                               newPoly.bbox.maxY < oldPoly.bbox.minY || 
                               newPoly.bbox.minY > oldPoly.bbox.maxY);
              return overlap;
            });
            
            if (matchingPrev && matchingPrev.settings) {
              // Preserve the previous polygon's settings but ALWAYS use new pixelBbox
              // The pixelBbox must match the current detection, not the old one
              return {
                ...newPoly,
                id: matchingPrev.id,
                settings: { ...matchingPrev.settings },
                needsReprocess: false
              };
            }
            return newPoly;
          });
        });
        
        // For backwards compatibility, also set the primary polygon (always use first on initial detect)
        if (refinedPolygons.length > 0) {
          const primary = refinedPolygons[0];
          setProcessedPath(primary.outer);
          setObjectDims({ 
            width: primary.bbox.maxX - primary.bbox.minX, 
            height: primary.bbox.maxY - primary.bbox.minY, 
            ...primary.bbox 
          });
          setDetectedShapeType(primary.type);
          setSelectedPolygonIndex(0);
        } else {
          setProcessedPath([]); setObjectDims(null); setDetectedShapeType(null);
          setDetectedPolygons([]);
        }
    } else {
        setProcessedPath([]); setObjectDims(null); setDetectedShapeType(null);
        setDetectedPolygons([]);
    }

    const endTime = performance.now();
    setDebugStats({ mappedPixels: mappedPixelCount, matrixValid: !isNaN(matrix[0]), processingTime: (endTime - startTime).toFixed(1) });
    
    if (segmentMode === 'auto' && (Math.abs(refR - calculatedRefColor.r) > 1 || Math.abs(refG - calculatedRefColor.g) > 1)) {
        setCalculatedRefColor({r: refR, g: refG, b: refB});
    }
    
    setIsProcessing(false);

  }, [imageSrc, corners, paperWidth, paperHeight, threshold, scanStep, curveSmoothing, imgDims, segmentMode, targetColor, selectionBox, showMask, invertResult, viewMode, calculatedRefColor, smartRefine, shadowRemoval, noiseFilter]);

  // Reprocess a single polygon - applies smoothing and shape fitting only
  const reprocessPolygon = useCallback((polygonIndex) => {
    if (polygonIndex < 0 || polygonIndex >= detectedPolygons.length) return;
    
    const poly = detectedPolygons[polygonIndex];
    if (!poly.settings) return;
    
    const { curveSmoothing: polyCurve, smartRefine: polySmartRefine } = poly.settings;
    
    // Use raw contour data if available
    let outerPoints = poly.rawOuter ? [...poly.rawOuter] : [...poly.outer];
    let detected = 'poly';
    
    // Apply smoothing and shape fitting
    if (polySmartRefine && outerPoints.length > 0) {
      const circleFit = ShapeFitter.fitCircle(outerPoints);
      if (circleFit) {
        outerPoints = ShapeFitter.generateCircle(circleFit.cx, circleFit.cy, circleFit.r);
        detected = 'circle';
      } else {
        outerPoints = VectorUtils.simplify(outerPoints, 0.5);
        outerPoints = VectorUtils.smooth(outerPoints, polyCurve);
      }
    } else if (outerPoints.length > 0) {
      outerPoints = VectorUtils.simplify(outerPoints, 0.5);
      outerPoints = VectorUtils.smooth(outerPoints, polyCurve);
    }
    
    // Process holes
    const rawHoles = poly.rawHoles || poly.holes;
    const refinedHoles = rawHoles.map(hole => {
      let holePoints = [...hole];
      if (polySmartRefine) {
        const holeFit = ShapeFitter.fitCircle(hole);
        if (holeFit) {
          holePoints = ShapeFitter.generateCircle(holeFit.cx, holeFit.cy, holeFit.r);
        } else {
          holePoints = VectorUtils.simplify(holePoints, 0.5);
          holePoints = VectorUtils.smooth(holePoints, polyCurve);
        }
      } else {
        holePoints = VectorUtils.simplify(holePoints, 0.5);
        holePoints = VectorUtils.smooth(holePoints, polyCurve);
      }
      return holePoints;
    });
    
    // Update the polygon
    setDetectedPolygons(prev => {
      const updated = [...prev];
      updated[polygonIndex] = {
        ...updated[polygonIndex],
        outer: outerPoints,
        holes: refinedHoles,
        type: detected,
        needsReprocess: false
      };
      return updated;
    });
    
    // Update primary display if this is the selected polygon
    if (polygonIndex === selectedPolygonIndex) {
      setProcessedPath(outerPoints);
      const minX = Math.min(...outerPoints.map(p => p.x));
      const maxX = Math.max(...outerPoints.map(p => p.x));
      const minY = Math.min(...outerPoints.map(p => p.y));
      const maxY = Math.max(...outerPoints.map(p => p.y));
      setObjectDims({ width: maxX - minX, height: maxY - minY, minX, maxX, minY, maxY });
      setDetectedShapeType(detected);
    }
  }, [detectedPolygons, selectedPolygonIndex]);
  
  // Reprocess polygon detection - regenerates mask and contours for a polygon's ROI with its settings
  const reprocessPolygonDetection = useCallback((polygonIndex) => {
    console.log('=== reprocessPolygonDetection START ===');
    console.log('Requested polygonIndex:', polygonIndex, 'Total polygons:', detectedPolygons.length);
    console.log('ALL polygon pixelBboxes:');
    detectedPolygons.forEach((p, i) => {
      console.log(`  Polygon ${i}: pixelBbox=`, p.pixelBbox, 'mm bbox=', p.bbox);
    });
    
    if (!unwarpedBufferRef.current || polygonIndex < 0 || polygonIndex >= detectedPolygons.length) {
      console.log('Early exit - invalid state');
      return;
    }
    
    const poly = detectedPolygons[polygonIndex];
    console.log('USING polygon:', polygonIndex, 'pixelBbox:', JSON.stringify(poly.pixelBbox), 'mm bbox:', JSON.stringify(poly.bbox));
    if (!poly.settings || !poly.pixelBbox) {
      console.log('Early exit - no settings or pixelBbox');
      return;
    }
    
    const { width: bufWidth, height: bufHeight, data: rawBuffer } = unwarpedBufferRef.current;
    const { threshold: polyThreshold, noiseFilter: polyNoise, shadowRemoval: polyShadow, scanStep: polyScan, curveSmoothing: polyCurve, smartRefine: polySmartRefine } = poly.settings;
    // Always use global invertResult - it's a fundamental detection mode, not per-polygon
    const polyInvert = invertResult;
    
    // Get reference color from state
    const refR = calculatedRefColor.r || 255;
    const refG = calculatedRefColor.g || 255;
    const refB = calculatedRefColor.b || 255;
    
    // Expand bounding box with margin for better edge detection
    const margin = 10;
    const roiMinX = Math.max(0, poly.pixelBbox.minX - margin);
    const roiMaxX = Math.min(bufWidth - 1, poly.pixelBbox.maxX + margin);
    const roiMinY = Math.max(0, poly.pixelBbox.minY - margin);
    const roiMaxY = Math.min(bufHeight - 1, poly.pixelBbox.maxY + margin);
    const roiWidth = roiMaxX - roiMinX + 1;
    const roiHeight = roiMaxY - roiMinY + 1;
    
    if (roiWidth < 5 || roiHeight < 5) return;
    
    // Create mask for ROI using polygon's settings
    const roiMask = new Uint8Array(roiWidth * roiHeight);
    
    for (let ry = 0; ry < roiHeight; ry++) {
      for (let rx = 0; rx < roiWidth; rx++) {
        const globalX = roiMinX + rx;
        const globalY = roiMinY + ry;
        const srcIdx = (globalY * bufWidth + globalX) * 4;
        
        const r = rawBuffer[srcIdx];
        const g = rawBuffer[srcIdx + 1];
        const b = rawBuffer[srcIdx + 2];
        
        // Apply noise filter if enabled
        let testR = r, testG = g, testB = b;
        if (polyNoise > 0 && globalX > 0 && globalX < bufWidth - 1 && globalY > 0 && globalY < bufHeight - 1) {
          const w4 = bufWidth * 4;
          testR = (r + rawBuffer[srcIdx-4] + rawBuffer[srcIdx+4] + rawBuffer[srcIdx-w4] + rawBuffer[srcIdx+w4]) / 5;
          testG = (g + rawBuffer[srcIdx+1-4] + rawBuffer[srcIdx+1+4] + rawBuffer[srcIdx+1-w4] + rawBuffer[srcIdx+1+w4]) / 5;
          testB = (b + rawBuffer[srcIdx+2-4] + rawBuffer[srcIdx+2+4] + rawBuffer[srcIdx+2-w4] + rawBuffer[srcIdx+2+w4]) / 5;
        }
        
        const dist = Math.sqrt((refR-testR)**2 + (refG-testG)**2 + (refB-testB)**2);
        let isObject = segmentMode === 'manual-obj' ? dist < polyThreshold : dist > polyThreshold;
        if (polyInvert) isObject = !isObject;
        
        roiMask[ry * roiWidth + rx] = isObject ? 1 : 0;
      }
    }
    
    // Apply morphological erosion (shadow removal)
    if (polyShadow > 0) {
      const erosionIterations = Math.floor(polyShadow);
      for (let iter = 0; iter < erosionIterations; iter++) {
        const erodedMask = new Uint8Array(roiWidth * roiHeight);
        for (let ry = 1; ry < roiHeight - 1; ry++) {
          for (let rx = 1; rx < roiWidth - 1; rx++) {
            const i = ry * roiWidth + rx;
            if (roiMask[i] === 1) {
              if (roiMask[i-1]===0 || roiMask[i+1]===0 || roiMask[i-roiWidth]===0 || roiMask[i+roiWidth]===0) {
                erodedMask[i] = 0;
              } else {
                erodedMask[i] = 1;
              }
            }
          }
        }
        roiMask.set(erodedMask);
      }
    }
    
    // Use ContourTracer's low-level functions directly to avoid coordinate conversion issues
    // Label connected components in ROI
    const { labels, components } = ContourTracer.labelComponents(roiMask, roiWidth, roiHeight);
    
    if (components.length === 0) return;
    
    // Get the largest component
    components.sort((a, b) => b.size - a.size);
    const mainComponent = components[0];
    
    // Find starting point for boundary tracing
    let startPixel = mainComponent.pixels[0];
    for (const p of mainComponent.pixels) {
      if (p.y < startPixel.y || (p.y === startPixel.y && p.x < startPixel.x)) {
        startPixel = p;
      }
    }
    
    // Trace boundary in ROI pixel coordinates
    const boundary = ContourTracer.traceBoundary4(roiMask, roiWidth, roiHeight, startPixel.x, startPixel.y, mainComponent.label, labels);
    if (boundary.length < 5) return;
    
    // Simplify contour
    const simplified = ContourTracer.simplifyContour(boundary, polyScan);
    
    // Helper to convert ROI pixel coords to global mm coords directly
    const roiPixelToGlobalMm = (p) => {
      // p.x, p.y are in ROI pixel space (0 to roiWidth-1, 0 to roiHeight-1)
      // Convert to global buffer pixel coords
      const globalPixelX = roiMinX + p.x;
      const globalPixelY = roiMinY + p.y;
      // Convert to global mm (Y is flipped: buffer Y=0 is top, mm Y=0 is bottom)
      return {
        x: (globalPixelX / bufWidth) * paperWidth,
        y: ((bufHeight - globalPixelY) / bufHeight) * paperHeight
      };
    };
    
    // Convert to mm and ensure CCW winding for outer contour
    let outerPoints = simplified.map(roiPixelToGlobalMm);
    if (ContourTracer.signedArea(outerPoints) < 0) {
      outerPoints = ContourTracer.reverseContour(outerPoints);
    }
    // Close the contour
    if (outerPoints.length > 0) {
      outerPoints.push({ ...outerPoints[0] });
    }
    
    const rawOuter = [...outerPoints];
    
    // Apply smoothing and shape fitting
    let detected = 'poly';
    if (polySmartRefine && outerPoints.length > 0) {
      const circleFit = ShapeFitter.fitCircle(outerPoints);
      if (circleFit) {
        outerPoints = ShapeFitter.generateCircle(circleFit.cx, circleFit.cy, circleFit.r);
        detected = 'circle';
      } else {
        outerPoints = VectorUtils.simplify(outerPoints, 0.5);
        outerPoints = VectorUtils.smooth(outerPoints, polyCurve);
      }
    } else if (outerPoints.length > 0) {
      outerPoints = VectorUtils.simplify(outerPoints, 0.5);
      outerPoints = VectorUtils.smooth(outerPoints, polyCurve);
    }
    
    // Find and process holes
    const holes = ContourTracer.findHoles(roiMask, labels, mainComponent.label, roiWidth, roiHeight, mainComponent.bbox);
    const refinedHoles = [];
    const rawHoles = [];
    
    for (const holePixels of holes) {
      if (holePixels.length < 5) continue;
      
      // Find starting point for hole
      let startHole = holePixels[0];
      for (const p of holePixels) {
        if (p.y < startHole.y || (p.y === startHole.y && p.x < startHole.x)) startHole = p;
      }
      
      // Create temporary mask for hole tracing
      const holeMask = new Uint8Array(roiWidth * roiHeight);
      holePixels.forEach(p => { holeMask[p.y * roiWidth + p.x] = 1; });
      
      const holeBoundary = ContourTracer.traceBoundary4(holeMask, roiWidth, roiHeight, startHole.x, startHole.y, null, null);
      if (holeBoundary.length < 5) continue;
      
      const holeSimplified = ContourTracer.simplifyContour(holeBoundary, polyScan);
      let holePoints = holeSimplified.map(roiPixelToGlobalMm);
      
      // Ensure CW winding for holes
      if (ContourTracer.signedArea(holePoints) > 0) {
        holePoints = ContourTracer.reverseContour(holePoints);
      }
      if (holePoints.length > 0) {
        holePoints.push({ ...holePoints[0] });
      }
      
      rawHoles.push([...holePoints]);
      
      // Apply refinement
      if (polySmartRefine) {
        const holeFit = ShapeFitter.fitCircle(holePoints);
        if (holeFit) {
          holePoints = ShapeFitter.generateCircle(holeFit.cx, holeFit.cy, holeFit.r);
        } else {
          holePoints = VectorUtils.simplify(holePoints, 0.5);
          holePoints = VectorUtils.smooth(holePoints, polyCurve);
        }
      } else {
        holePoints = VectorUtils.simplify(holePoints, 0.5);
        holePoints = VectorUtils.smooth(holePoints, polyCurve);
      }
      refinedHoles.push(holePoints);
    }
    
    // Update the polygon with new detection results
    setDetectedPolygons(prev => {
      const updated = [...prev];
      const minX = Math.min(...outerPoints.map(p => p.x));
      const maxX = Math.max(...outerPoints.map(p => p.x));
      const minY = Math.min(...outerPoints.map(p => p.y));
      const maxY = Math.max(...outerPoints.map(p => p.y));
      
      // Recalculate pixelBbox for subsequent edits
      const newPixelBbox = {
        minX: Math.floor((minX / paperWidth) * bufWidth),
        maxX: Math.ceil((maxX / paperWidth) * bufWidth),
        minY: Math.floor((1 - maxY / paperHeight) * bufHeight),
        maxY: Math.ceil((1 - minY / paperHeight) * bufHeight)
      };
      
      updated[polygonIndex] = {
        ...updated[polygonIndex],
        outer: outerPoints,
        holes: refinedHoles,
        rawOuter,
        rawHoles,
        type: detected,
        bbox: { minX, maxX, minY, maxY },
        pixelBbox: newPixelBbox,
        needsReprocess: false,
        needsDetectionReprocess: false
      };
      return updated;
    });
    
    // Update primary display if this is the selected polygon
    if (polygonIndex === selectedPolygonIndex) {
      setProcessedPath(outerPoints);
      const minX = Math.min(...outerPoints.map(p => p.x));
      const maxX = Math.max(...outerPoints.map(p => p.x));
      const minY = Math.min(...outerPoints.map(p => p.y));
      const maxY = Math.max(...outerPoints.map(p => p.y));
      setObjectDims({ width: maxX - minX, height: maxY - minY, minX, maxX, minY, maxY });
      setDetectedShapeType(detected);
    }
  }, [detectedPolygons, selectedPolygonIndex, calculatedRefColor, segmentMode, paperWidth, paperHeight, invertResult]);
  
  // Effect to reprocess polygons when their settings change
  useEffect(() => {
    // Check for detection reprocess first (higher priority)
    // Prioritize the selected polygon if it needs reprocessing
    let polygonNeedsDetection = -1;
    if (detectedPolygons[selectedPolygonIndex]?.needsDetectionReprocess) {
      polygonNeedsDetection = selectedPolygonIndex;
      console.log('Selected polygon needs detection reprocess:', selectedPolygonIndex);
    } else {
      polygonNeedsDetection = detectedPolygons.findIndex(p => p.needsDetectionReprocess);
      if (polygonNeedsDetection !== -1) {
        console.log('Found polygon needing detection reprocess via findIndex:', polygonNeedsDetection);
      }
    }
    
    if (polygonNeedsDetection !== -1) {
      console.log('Scheduling reprocessPolygonDetection for polygon:', polygonNeedsDetection);
      const idx = polygonNeedsDetection; // Capture in closure
      const timer = setTimeout(() => {
        console.log('Executing reprocessPolygonDetection for polygon:', idx);
        reprocessPolygonDetection(idx);
      }, 150);
      return () => clearTimeout(timer);
    }
    
    // Then check for vector-only reprocess - prioritize selected polygon
    let polygonToReprocess = -1;
    if (detectedPolygons[selectedPolygonIndex]?.needsReprocess) {
      polygonToReprocess = selectedPolygonIndex;
    } else {
      polygonToReprocess = detectedPolygons.findIndex(p => p.needsReprocess);
    }
    
    if (polygonToReprocess !== -1) {
      const idx = polygonToReprocess;
      const timer = setTimeout(() => reprocessPolygon(idx), 100);
      return () => clearTimeout(timer);
    }
  }, [detectedPolygons, reprocessPolygon, reprocessPolygonDetection, selectedPolygonIndex]);

  // Effect to update display when polygon selection changes (without reprocessing)
  useEffect(() => {
    if (detectedPolygons.length > 0 && detectedPolygons[selectedPolygonIndex]) {
      const poly = detectedPolygons[selectedPolygonIndex];
      setProcessedPath(poly.outer);
      if (poly.outer && poly.outer.length > 0) {
        const minX = Math.min(...poly.outer.map(p => p.x));
        const maxX = Math.max(...poly.outer.map(p => p.x));
        const minY = Math.min(...poly.outer.map(p => p.y));
        const maxY = Math.max(...poly.outer.map(p => p.y));
        setObjectDims({ width: maxX - minX, height: maxY - minY, minX, maxX, minY, maxY });
      }
      setDetectedShapeType(poly.type);
    }
  }, [selectedPolygonIndex, detectedPolygons]);

  useEffect(() => {
    if (step === 'process') {
      const timer = setTimeout(processImage, 50); 
      return () => clearTimeout(timer);
    }
  }, [step, threshold, scanStep, curveSmoothing, processImage, segmentMode, showMask, invertResult, viewMode, smartRefine, shadowRemoval, noiseFilter]);

  const openSaveDialog = (type) => {
    const defaultName = (type === 'dxf' || type === 'svg') ? 'scan_shape' : `scan_w${paperWidth}mm_h${paperHeight}mm`;
    setSaveFileName(defaultName);
    setSaveType(type);
    setShowSaveDialog(true);
  };

  const confirmSave = async () => {
    const cleanName = saveFileName.trim() || 'scan_shape';
    setShowSaveDialog(false);
    if (saveType === 'dxf') {
      await performDXFSave(cleanName);
    } else if (saveType === 'svg') {
      await performSVGSave(cleanName);
    } else {
      await performImageSave(cleanName);
    }
  };

  const performDXFSave = async (baseName) => {
    if (detectedPolygons.length === 0 && processedPath.length < 2) return;
    
    // DXF Header with units set to millimeters ($INSUNITS = 4)
    let dxf = "0\nSECTION\n2\nHEADER\n";
    dxf += "9\n$ACADVER\n1\nAC1014\n"; // AutoCAD R14 format
    dxf += "9\n$INSUNITS\n70\n4\n"; // 4 = Millimeters
    dxf += "9\n$MEASUREMENT\n70\n1\n"; // 1 = Metric
    dxf += "9\n$LUNITS\n70\n2\n"; // 2 = Decimal
    dxf += "9\n$LUPREC\n70\n3\n"; // 3 decimal places
    dxf += "0\nENDSEC\n";
    
    // Tables section (required by some CAD programs)
    dxf += "0\nSECTION\n2\nTABLES\n";
    dxf += "0\nTABLE\n2\nLAYER\n70\n1\n";
    dxf += "0\nLAYER\n2\n0\n70\n0\n62\n7\n6\nCONTINUOUS\n";
    dxf += "0\nENDTAB\n";
    dxf += "0\nENDSEC\n";
    
    // Entities section
    dxf += "0\nSECTION\n2\nENTITIES\n";
    
    // Export all detected polygons with their holes
    if (detectedPolygons.length > 0) {
      detectedPolygons.forEach((poly, polyIdx) => {
        // Export outer contour
        if (poly.outer.length >= 2) {
          dxf += "0\nLWPOLYLINE\n8\nShape_" + (polyIdx + 1) + "_Outer\n90\n" + poly.outer.length + "\n70\n1\n";
          poly.outer.forEach(p => { dxf += "10\n" + p.x.toFixed(3) + "\n20\n" + p.y.toFixed(3) + "\n"; });
        }
        
        // Export holes (only if showHoles is enabled for this polygon)
        if (poly.settings?.showHoles !== false) {
          poly.holes.forEach((hole, holeIdx) => {
            if (hole.length >= 2) {
              dxf += "0\nLWPOLYLINE\n8\nShape_" + (polyIdx + 1) + "_Hole_" + (holeIdx + 1) + "\n90\n" + hole.length + "\n70\n1\n";
              hole.forEach(p => { dxf += "10\n" + p.x.toFixed(3) + "\n20\n" + p.y.toFixed(3) + "\n"; });
            }
          });
        }
      });
    } else if (processedPath.length >= 2) {
      // Fallback to single path
      dxf += "0\nLWPOLYLINE\n8\nObjectLayer\n90\n" + processedPath.length + "\n70\n1\n";
      processedPath.forEach(p => { dxf += "10\n" + p.x.toFixed(3) + "\n20\n" + p.y.toFixed(3) + "\n"; });
    }
    
    dxf += "0\nENDSEC\n0\nEOF";
    
    const filename = `${baseName}.dxf`;
    
    if (Capacitor.isNativePlatform()) {
      try {
        await Filesystem.requestPermissions();
        await Filesystem.writeFile({
          path: filename,
          data: dxf,
          directory: Directory.Documents,
          encoding: Encoding.UTF8,
        });
        alert(`DXF saved to Documents folder:\n${filename}`);
      } catch (error) {
        console.error('Error saving DXF:', error);
        alert('Error saving file. Please check storage permissions.');
      }
    } else {
      const blob = new Blob([dxf], { type: 'application/dxf' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // --- SVG Export ---
  const performSVGSave = async (baseName) => {
    if (detectedPolygons.length === 0 && processedPath.length < 2) return;
    
    // SVG with proper dimensions in mm
    let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
    svg += `<svg xmlns="http://www.w3.org/2000/svg" width="${paperWidth}mm" height="${paperHeight}mm" viewBox="0 0 ${paperWidth} ${paperHeight}">\n`;
    svg += `  <title>ShapeScanner Export - ${paperWidth}mm x ${paperHeight}mm</title>\n`;
    svg += `  <desc>Exported from ShapeScanner. All dimensions in millimeters.</desc>\n`;
    
    if (detectedPolygons.length > 0) {
      detectedPolygons.forEach((poly, polyIdx) => {
        // Outer contour
        if (poly.outer.length >= 2) {
          const pathData = poly.outer.map((p, i) => 
            `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(3)} ${(paperHeight - p.y).toFixed(3)}`
          ).join(' ') + ' Z';
          svg += `  <path id="shape_${polyIdx + 1}_outer" d="${pathData}" fill="none" stroke="#000000" stroke-width="0.1"/>\n`;
        }
        // Holes (only if showHoles is enabled for this polygon)
        if (poly.settings?.showHoles !== false) {
          poly.holes.forEach((hole, holeIdx) => {
            if (hole.length >= 2) {
              const holeData = hole.map((p, i) => 
                `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(3)} ${(paperHeight - p.y).toFixed(3)}`
              ).join(' ') + ' Z';
              svg += `  <path id="shape_${polyIdx + 1}_hole_${holeIdx + 1}" d="${holeData}" fill="none" stroke="#666666" stroke-width="0.1"/>\n`;
            }
          });
        }
      });
    } else if (processedPath.length >= 2) {
      const pathData = processedPath.map((p, i) => 
        `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(3)} ${(paperHeight - p.y).toFixed(3)}`
      ).join(' ') + ' Z';
      svg += `  <path id="shape" d="${pathData}" fill="none" stroke="#000000" stroke-width="0.1"/>\n`;
    }
    
    svg += `</svg>`;
    
    const filename = `${baseName}.svg`;
    
    if (Capacitor.isNativePlatform()) {
      try {
        await Filesystem.requestPermissions();
        await Filesystem.writeFile({
          path: filename,
          data: svg,
          directory: Directory.Documents,
          encoding: Encoding.UTF8,
        });
        alert(`SVG saved to Documents folder:\n${filename}`);
      } catch (error) {
        console.error('Error saving SVG:', error);
        alert('Error saving file. Please check storage permissions.');
      }
    } else {
      const blob = new Blob([svg], { type: 'image/svg+xml' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // --- NEW: Export Image for CAD ---
  const performImageSave = async (baseName) => {
      if (!sourcePixelData.current || !unwarpedBufferRef.current) return;
      const { width, height, data } = unwarpedBufferRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(width, height);
      for(let i=0; i<data.length; i++) imgData.data[i] = data[i];
      ctx.putImageData(imgData, 0, 0);
      
      const filename = `${baseName}.png`;
      
      if (Capacitor.isNativePlatform()) {
        try {
          await Filesystem.requestPermissions();
          const base64Data = canvas.toDataURL('image/png').split(',')[1];
          await Filesystem.writeFile({
            path: filename,
            data: base64Data,
            directory: Directory.Documents,
          });
          alert(`Image saved to Documents folder:\n${filename}`);
        } catch (error) {
          console.error('Error saving image:', error);
          alert('Error saving file. Please check storage permissions.');
        }
      } else {
        const link = document.createElement('a');
        link.download = filename;
        link.href = canvas.toDataURL('image/png');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
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
    <div data-theme={isDarkTheme ? 'dark' : 'light'} className="fixed inset-0 theme-bg-primary theme-text-primary font-sans overflow-hidden touch-none select-none h-[100dvh] flex flex-col">
      
      {/* Header */}
      <div className="flex items-center justify-between p-4 theme-bg-secondary border-b theme-border z-10 shrink-0">
        <div className="flex items-center gap-2">
           <RefreshCcw className="text-[var(--accent-blue)]" size={20} onClick={() => setStep('capture')}/>
           <h1 className="font-bold text-lg tracking-tight theme-text-primary">ShapeScanner</h1>
        </div>
        <div className="flex items-center gap-2">
            <button 
                onClick={() => setShowSettingsMenu(true)} 
                className="p-1.5 rounded theme-text-secondary hover:theme-text-primary transition-colors"
                title="Settings"
            >
                <Settings size={18} />
            </button>
            <button 
                onClick={() => setShowDebug(!showDebug)} 
                className={`p-1 rounded ${showDebug ? 'bg-red-500 text-white' : 'theme-text-muted hover:theme-text-primary'}`}
                title="Debug Info"
            >
                <Bug size={16} />
            </button>
            <div className="text-xs font-mono theme-bg-tertiary theme-text-secondary px-2 py-1 rounded">
                {step.toUpperCase()}
            </div>
        </div>
      </div>

      {/* Tutorial Overlay */}
      {showTutorial && (
        <TutorialOverlay
          currentStep={tutorialStep}
          totalSteps={TUTORIAL_STEPS.length}
          stepData={TUTORIAL_STEPS[tutorialStep]}
          onNext={() => setTutorialStep(prev => Math.min(prev + 1, TUTORIAL_STEPS.length - 1))}
          onPrev={() => setTutorialStep(prev => Math.max(prev - 1, 0))}
          onClose={handleTutorialClose}
          onDontShowAgain={handleDontShowAgain}
        />
      )}

      {/* Settings Menu */}
      <SettingsMenu
        isOpen={showSettingsMenu}
        onClose={() => setShowSettingsMenu(false)}
        showTutorialOnStart={showTutorialOnStart}
        setShowTutorialOnStart={setShowTutorialOnStart}
        onOpenTutorial={openTutorial}
        isDarkTheme={isDarkTheme}
        setIsDarkTheme={setIsDarkTheme}
      />

      {/* Save Dialog Modal */}
      {showSaveDialog && (
        <div className="fixed inset-0 z-[100] bg-[var(--bg-overlay)] backdrop-blur-sm flex items-center justify-center p-4">
          <div className="theme-bg-secondary rounded-2xl border theme-border p-6 w-full max-w-sm shadow-2xl">
            <h3 className="theme-text-primary font-bold text-lg mb-4">Save {saveType === 'dxf' ? 'DXF Vector' : saveType === 'svg' ? 'SVG Vector' : 'Image'}</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs theme-text-secondary uppercase font-bold block mb-2">File Name</label>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={saveFileName}
                    onChange={(e) => setSaveFileName(e.target.value)}
                    placeholder="Enter filename"
                    className="flex-1 theme-bg-input border theme-border rounded-lg px-4 py-3 theme-text-primary font-mono text-sm focus:border-blue-500 focus:outline-none transition-colors"
                    autoFocus
                  />
                  <span className="theme-text-muted font-mono text-sm">.{saveType === 'dxf' ? 'dxf' : saveType === 'svg' ? 'svg' : 'png'}</span>
                </div>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="flex-1 theme-bg-tertiary hover:opacity-80 border theme-border theme-text-primary font-bold py-3 rounded-xl transition-all"
              >
                Cancel
              </button>
              <button
                onClick={confirmSave}
                className="flex-1 bg-[var(--accent-emerald)] hover:bg-[var(--accent-emerald-hover)] text-white font-bold py-3 rounded-xl flex items-center justify-center gap-2 transition-all"
              >
                <Download size={16} /> Save
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 relative theme-bg-primary overflow-hidden w-full h-full flex flex-col">
        
        {/* DEBUG OVERLAY */}
        {showDebug && (
            <div className="absolute top-0 left-0 bg-[var(--bg-overlay)] text-green-400 p-2 text-[10px] font-mono z-50 pointer-events-none border theme-border m-2 rounded">
                <p>Img: {imgDims.w}x{imgDims.h}</p>
                <p>Target: {Math.floor(paperWidth*2)}x{Math.floor(paperHeight*2)}</p>
                <p>Mapped: {debugStats.mappedPixels} px</p>
                <p>Time: {debugStats.processingTime}ms</p>
                <p>Mode: {viewMode}</p>
                <p>Shapes: {detectedPolygons.length} | Holes: {detectedPolygons.reduce((a,p)=>a+p.holes.length,0)}</p>
            </div>
        )}

        {/* STEP 1: CAPTURE */}
        {step === 'capture' && (
          <div className="h-full flex flex-col items-center justify-center p-6 space-y-8 theme-bg-primary w-full">
            <div className="text-center space-y-4">
              <div className="w-24 h-32 border-2 border-dashed theme-border mx-auto theme-bg-secondary rounded-lg flex items-center justify-center shadow-[0_0_20px_rgba(0,0,0,0.3)]">
                 <div className="w-10 h-10 theme-bg-tertiary rounded-full flex items-center justify-center">
                    <ScanLine size={20} className="theme-text-muted"/>
                 </div>
              </div>
              <div className="space-y-1">
                  <h2 className="theme-text-primary font-bold text-lg">Scan Object</h2>
                  <p className="theme-text-muted text-sm max-w-[220px] mx-auto">
                    Place object on a white sheet. Ensure all 4 corners are visible.
                  </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 w-full max-w-xs">
                <button onClick={() => fileInputRef.current.click()} className="flex flex-col items-center justify-center bg-[var(--accent-blue)] hover:bg-[var(--accent-blue-hover)] active:scale-95 transition-all p-6 rounded-2xl shadow-lg group text-white">
                    <Upload size={32} className="mb-2 group-hover:-translate-y-1 transition-transform"/><span className="font-bold">Upload</span>
                </button>
                <button onClick={handleCameraCapture} className="flex flex-col items-center justify-center theme-bg-tertiary hover:opacity-80 active:scale-95 transition-all p-6 rounded-2xl border theme-border group theme-text-primary">
                    <CameraIcon size={32} className="mb-2 group-hover:-translate-y-1 transition-transform"/><span className="font-bold">Camera</span>
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
                <div className="bg-[var(--bg-overlay)] backdrop-blur rounded-lg p-1 pointer-events-auto shadow-lg flex flex-col gap-1 border theme-border">
                    <button onClick={() => setView(v => ({...v, scale: v.scale * 1.2}))} className="p-2 hover:opacity-70 rounded theme-text-secondary"><ZoomIn size={20}/></button>
                    <button onClick={() => setView(v => ({...v, scale: v.scale * 0.8}))} className="p-2 hover:opacity-70 rounded theme-text-secondary"><ZoomOut size={20}/></button>
                    <button onClick={fitToScreen} className="p-2 hover:opacity-70 rounded text-[var(--accent-blue)]"><Maximize2 size={20}/></button>
                </div>
                </div>
                
                <div className="absolute top-4 left-4 pointer-events-none flex flex-col gap-4">
                    <button 
                        onClick={autoDetectCorners}
                        className="pointer-events-auto bg-[var(--accent-blue)]/90 hover:bg-[var(--accent-blue)] backdrop-blur text-white px-4 py-2 rounded-full text-xs font-bold shadow-lg flex items-center gap-2 w-max transition-colors"
                    >
                        <ScanLine size={14} /> Auto-Detect
                    </button>

                    <div className="bg-[var(--bg-overlay)] backdrop-blur rounded-lg p-3 pointer-events-auto shadow-lg flex flex-col gap-3 border theme-border w-44">
                        <div className="flex items-center justify-between">
                            <span className="text-xs font-bold theme-text-secondary flex items-center gap-2"><Pipette size={12}/> Paper</span>
                            <div className="flex items-center gap-2">
                                <div 
                                    className="w-5 h-5 rounded-full border theme-border shadow-inner" 
                                    style={{backgroundColor: `rgb(${paperColor.r},${paperColor.g},${paperColor.b})`}} 
                                />
                                <button 
                                    onClick={() => setIsPickingPaperColor(!isPickingPaperColor)}
                                    className={`px-2 py-1 rounded text-[10px] font-bold uppercase transition-all ${isPickingPaperColor ? 'bg-[var(--accent-amber)] text-black' : 'theme-bg-tertiary theme-text-secondary hover:opacity-70'}`}
                                >
                                    {isPickingPaperColor ? 'Tap Paper' : 'Pick'}
                                </button>
                            </div>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-xs font-bold theme-text-secondary flex items-center gap-2"><Palette size={12}/> B&W</span>
                            <button 
                                onClick={() => setCalMonochrome(!calMonochrome)}
                                className={`w-8 h-4 rounded-full relative transition-colors ${calMonochrome ? 'bg-[var(--accent-blue)]' : 'bg-[var(--bg-tertiary)]'}`}
                            >
                                <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${calMonochrome ? 'left-4.5' : 'left-0.5'}`} />
                            </button>
                        </div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-xs theme-text-secondary">
                                <span className="flex items-center gap-1"><Sun size={12} /> Contrast</span>
                                <span>{calContrast}%</span>
                            </div>
                            <input 
                                type="range" min="25" max="300" value={calContrast} 
                                onChange={(e) => setCalContrast(Number(e.target.value))}
                                className="w-full h-1.5 bg-[var(--bg-tertiary)] rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                        </div>
                    </div>
                    
                    {isPickingPaperColor && (
                        <div className="bg-[var(--accent-amber)]/90 backdrop-blur text-black font-bold px-4 py-2 rounded-full text-xs shadow-lg animate-pulse pointer-events-none">
                            Tap on the paper
                        </div>
                    )}
                </div>
            </div>

            <div className="w-full theme-bg-secondary p-4 rounded-t-2xl shadow-[0_-4px_20px_rgba(0,0,0,0.3)] border-t theme-border shrink-0 z-20">
                <div className="flex gap-3 mb-4">
                    <div className="flex flex-col gap-1 w-24 shrink-0">
                        <label className="text-[10px] theme-text-muted uppercase font-bold flex justify-between items-center mb-1">
                            Format
                            <button onClick={toggleOrientation} className="theme-text-secondary hover:theme-text-primary transition-colors theme-bg-tertiary p-1 rounded"><RotateCcw size={10} /></button>
                        </label>
                        <div className="flex flex-col gap-1.5">
                            <button onClick={() => applyPreset(210, 297)} className="theme-bg-tertiary hover:opacity-80 border theme-border px-2 py-1.5 rounded text-xs flex items-center gap-2 theme-text-secondary transition-colors"><FileText size={12}/> A4</button>
                            <button onClick={() => applyPreset(215.9, 279.4)} className="theme-bg-tertiary hover:opacity-80 border theme-border px-2 py-1.5 rounded text-xs flex items-center gap-2 theme-text-secondary transition-colors"><FileText size={12}/> Letter</button>
                            <button onClick={() => applyPreset(85.6, 53.98)} className="theme-bg-tertiary hover:opacity-80 border theme-border px-2 py-1.5 rounded text-xs flex items-center gap-2 theme-text-secondary transition-colors"><CreditCard size={12}/> Card</button>
                        </div>
                    </div>

                    <div className="flex-1 flex gap-3">
                        <div className="flex-1">
                            <label className="text-[10px] theme-text-muted uppercase font-bold mb-1 block">Width (mm)</label>
                            <input type="number" value={paperWidth} onChange={e => setPaperWidth(Number(e.target.value))} className="w-full theme-bg-input border theme-border rounded-lg p-3 theme-text-primary font-mono text-sm focus:border-blue-500 focus:outline-none transition-colors"/>
                        </div>
                        <div className="flex-1">
                            <label className="text-[10px] theme-text-muted uppercase font-bold mb-1 block">Height (mm)</label>
                            <input type="number" value={paperHeight} onChange={e => setPaperHeight(Number(e.target.value))} className="w-full theme-bg-input border theme-border rounded-lg p-3 theme-text-primary font-mono text-sm focus:border-blue-500 focus:outline-none transition-colors"/>
                        </div>
                    </div>
                </div>
                <button onClick={() => setStep('process')} className="w-full bg-[var(--accent-blue)] hover:bg-[var(--accent-blue-hover)] text-white font-bold py-3.5 rounded-xl flex items-center justify-center gap-2 transition-all active:scale-[0.98] shadow-lg shadow-blue-900/20">
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
                <div className="absolute inset-0 z-[60] bg-[var(--bg-overlay)] backdrop-blur-sm flex justify-end">
                    <div className="w-full max-w-sm h-full theme-bg-secondary border-l theme-border shadow-2xl flex flex-col animate-[slideInRight_0.3s_ease-out]">
                        <div className="flex items-center justify-between p-4 border-b theme-border">
                            <h3 className="font-bold theme-text-primary flex items-center gap-2">
                                <Sparkles size={18} className="text-purple-500" /> AI Analyst
                            </h3>
                            <button onClick={() => setShowAiPanel(false)} className="p-2 hover:theme-bg-tertiary rounded theme-text-secondary">
                                <X size={20} />
                            </button>
                        </div>
                        <div className="flex-1 p-6 overflow-y-auto">
                            {isAiLoading ? (
                                <div className="flex flex-col items-center justify-center h-full space-y-4">
                                    <div className="w-12 h-12 border-4 border-purple-600/30 border-t-purple-600 rounded-full animate-spin"></div>
                                    <p className="theme-text-secondary text-sm animate-pulse">Analyzing geometry...</p>
                                </div>
                            ) : (
                                <div className="space-y-6">
                                    <div className="theme-bg-tertiary p-4 rounded-xl border theme-border">
                                        <h4 className="text-xs font-bold theme-text-muted uppercase mb-2">Detected Part</h4>
                                        <div className="theme-text-primary text-sm leading-relaxed whitespace-pre-line">{aiResult}</div>
                                    </div>
                                    <div className="text-xs theme-text-muted text-center">
                                        AI analysis based on {processedPath.length} vector points.
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            <div className="flex-1 min-h-0 flex flex-col w-full theme-bg-primary relative">
                {/* Process Toolbar */}
                <div className="w-full text-center text-xs theme-text-secondary p-2 flex justify-between items-center flex-wrap gap-2 shrink-0 z-10">
                    <div className="flex items-center gap-2">
                         <div className="flex theme-bg-secondary rounded-lg p-1 border theme-border">
                             <button 
                                onClick={() => setViewMode('original')}
                                className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 text-[10px] uppercase font-bold transition-all ${viewMode === 'original' ? 'theme-bg-tertiary theme-text-primary shadow-sm' : 'theme-text-muted hover:theme-text-secondary'}`}
                             >
                                <ImageIcon size={12}/> Orig
                             </button>
                             <button 
                                onClick={() => setViewMode('heatmap')}
                                className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 text-[10px] uppercase font-bold transition-all ${viewMode === 'heatmap' ? 'bg-orange-900/80 text-orange-200 shadow-sm' : 'theme-text-muted hover:theme-text-secondary'}`}
                             >
                                <Flame size={12}/> Heat
                             </button>
                             <button 
                                onClick={() => setViewMode('processed')}
                                className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 text-[10px] uppercase font-bold transition-all ${viewMode === 'processed' ? 'bg-emerald-900/80 text-emerald-200 shadow-sm' : 'theme-text-muted hover:theme-text-secondary'}`}
                             >
                                <Layers size={12}/> Proc
                             </button>
                             <button 
                                onClick={() => setViewMode('contour')}
                                className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 text-[10px] uppercase font-bold transition-all ${viewMode === 'contour' ? 'bg-purple-900/80 text-purple-200 shadow-sm' : 'theme-text-muted hover:theme-text-secondary'}`}
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
                                    : 'theme-bg-tertiary theme-border theme-text-secondary hover:opacity-80'
                            }`}
                         >
                            <Pipette size={12}/> {isPicking ? 'Set' : 'Ref'}
                        </button>
                    </div>
                </div>
                
                {/* Main Canvas Area - Centered and Scaled */}
                <div 
                    className="flex-1 w-full relative flex items-center justify-center overflow-hidden p-2"
                    onWheel={handleProcessViewWheel}
                >
                    <div 
                        className={`relative border theme-border shadow-2xl transition-colors duration-300 ${isPicking ? 'cursor-crosshair ring-2 ring-amber-500/50' : ''} ${isProcessPanning ? 'cursor-grabbing' : ''}`}
                        style={{
                            aspectRatio: `${paperWidth}/${paperHeight}`,
                            maxHeight: '100%',
                            maxWidth: '100%',
                            backgroundColor: viewMode === 'contour' ? 'white' : '#111', 
                            backgroundImage: viewMode === 'contour' ? 'none' : 'radial-gradient(#333 1px, transparent 1px)',
                            backgroundSize: '20px 20px',
                            transform: `translate(${processView.x}px, ${processView.y}px) scale(${processView.scale})`,
                            transformOrigin: 'center center'
                        }}
                    >
                        {!sourcePixelData.current && (
                            <div className="absolute inset-0 flex items-center justify-center bg-[var(--overlay-dark)] text-white flex-col z-50">
                                <AlertTriangle className="text-amber-500 mb-3" size={48} />
                                <span className="text-xl font-bold mb-1">Session Expired</span>
                                <button onClick={() => setStep('capture')} className="px-6 py-2 bg-[var(--accent-blue)] rounded-full font-bold text-sm mt-4 hover:bg-[var(--accent-blue-hover)] transition-colors">Reload Image</button>
                            </div>
                        )}
                        <canvas 
                            ref={processCanvasRef} 
                            className={`w-full h-full object-contain touch-none relative z-10 ${isProcessPanning ? 'cursor-grabbing' : ''}`}
                            onMouseDown={(e) => {
                                if (e.button === 1 || e.altKey) {
                                    e.preventDefault();
                                    handleProcessPanStart(e);
                                } else {
                                    handleProcessStart(e);
                                }
                            }}
                            onMouseMove={(e) => {
                                if (isProcessPanning) {
                                    handleProcessPanMove(e);
                                } else {
                                    handleProcessMove(e);
                                }
                            }}
                            onMouseUp={() => {
                                if (isProcessPanning) {
                                    handleProcessPanEnd();
                                } else {
                                    handleProcessEnd();
                                }
                            }}
                            onTouchStart={(e) => { const t = e.touches[0]; handleProcessStart({ clientX: t.clientX, clientY: t.clientY }); }}
                            onTouchMove={(e) => { const t = e.touches[0]; handleProcessMove({ clientX: t.clientX, clientY: t.clientY }); }}
                            onTouchEnd={handleProcessEnd}
                        />
                        
                        {(viewMode === 'processed' || viewMode === 'contour') && (
                            <svg className="absolute top-0 left-0 w-full h-full pointer-events-none z-20" viewBox={`0 0 100 100`} preserveAspectRatio="none">
                                {/* Render all detected polygons */}
                                {detectedPolygons.map((poly, polyIdx) => (
                                    <g key={polyIdx}>
                                        {/* Outer contour */}
                                        {poly.outer.length > 0 && (
                                            <path 
                                                d={`M ${poly.outer.map(p => `${(p.x/paperWidth)*100} ${(1 - p.y/paperHeight)*100}`).join(" L ")} Z`}
                                                fill="none" 
                                                stroke={viewMode === 'contour' ? '#000000' : (polyIdx === selectedPolygonIndex ? '#10b981' : '#3b82f6')} 
                                                strokeWidth={viewMode === 'contour' ? "1.5" : (polyIdx === selectedPolygonIndex ? "1.5" : "0.8")}
                                                vectorEffect="non-scaling-stroke"
                                            />
                                        )}
                                        {/* Holes */}
                                        {(poly.settings?.showHoles !== false) && poly.holes.map((hole, holeIdx) => (
                                            hole.length > 0 && (
                                                <path 
                                                    key={`hole-${holeIdx}`}
                                                    d={`M ${hole.map(p => `${(p.x/paperWidth)*100} ${(1 - p.y/paperHeight)*100}`).join(" L ")} Z`}
                                                    fill="none" 
                                                    stroke={viewMode === 'contour' ? '#666666' : '#f59e0b'} 
                                                    strokeWidth={viewMode === 'contour' ? "1" : "0.8"}
                                                    strokeDasharray={viewMode === 'contour' ? "none" : "2 1"}
                                                    vectorEffect="non-scaling-stroke"
                                                />
                                            )
                                        ))}
                                    </g>
                                ))}
                                
                                {/* Fallback to single processedPath if no polygons */}
                                {detectedPolygons.length === 0 && processedPath.length > 0 && (
                                    <path 
                                        d={`M ${processedPath.map(p => `${(p.x/paperWidth)*100} ${(1 - p.y/paperHeight)*100}`).join(" L ")} Z`}
                                        fill="none" 
                                        stroke={viewMode === 'contour' ? '#000000' : '#10b981'} 
                                        strokeWidth={viewMode === 'contour' ? "1.5" : "1"}
                                        vectorEffect="non-scaling-stroke"
                                    />
                                )}
                                
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
                            </svg>
                        )}
                        
                        {isPicking && (
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-[var(--accent-amber)]/90 backdrop-blur text-black font-bold px-6 py-3 rounded-full shadow-2xl pointer-events-none text-sm animate-bounce z-50 border-2 border-white">
                                Tap Background Color
                            </div>
                        )}
                        
                        {detectedPolygons.length > 0 && (
                            <div 
                                className="fixed flex flex-col gap-2 z-[100] cursor-grab active:cursor-grabbing select-none touch-none"
                                style={{
                                    left: badgePosition.x !== null ? `${badgePosition.x}px` : 'auto',
                                    right: badgePosition.x === null ? '16px' : 'auto',
                                    top: `${badgePosition.y}px`,
                                }}
                                onMouseDown={handleBadgeDragStart}
                                onTouchStart={handleBadgeDragStart}
                                onDoubleClick={resetBadgePosition}
                            >
                                <div className="bg-green-600/90 backdrop-blur text-white px-3 py-1.5 rounded-full shadow-lg text-[10px] font-bold uppercase tracking-widest flex items-center gap-1.5 border border-green-500">
                                    <Move size={10} className="opacity-60 mr-1" />
                                    <Check size={12} className="stroke-[3]" /> 
                                    {detectedPolygons.length} Shape{detectedPolygons.length > 1 ? 's' : ''} 
                                    {detectedPolygons.reduce((acc, p) => acc + (p.settings?.showHoles !== false ? p.holes.length : 0), 0) > 0 && (
                                        <span className="ml-1 text-amber-300">
                                            + {detectedPolygons.reduce((acc, p) => acc + (p.settings?.showHoles !== false ? p.holes.length : 0), 0)} Hole{detectedPolygons.reduce((acc, p) => acc + (p.settings?.showHoles !== false ? p.holes.length : 0), 0) > 1 ? 's' : ''}
                                        </span>
                                    )}
                                </div>
                                {detectedPolygons.length > 1 && (
                                    <div className="flex gap-1 flex-wrap">
                                        {detectedPolygons.map((_, idx) => (
                                            <button 
                                                key={idx}
                                                onClick={(e) => { e.stopPropagation(); setSelectedPolygonIndex(idx); }}
                                                className={`w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold transition-all ${
                                                    idx === selectedPolygonIndex 
                                                        ? 'bg-[var(--accent-emerald)] text-white shadow-lg' 
                                                        : 'bg-[var(--bg-overlay)] theme-text-secondary hover:opacity-80'
                                                }`}
                                            >
                                                {idx + 1}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                        
                        {/* Zoom controls for process view */}
                        {processView.scale !== 1 && (
                            <div className="absolute bottom-4 right-4 z-50 flex gap-2">
                                <button 
                                    onClick={resetProcessView}
                                    className="w-8 h-8 rounded-full theme-bg-tertiary border theme-border flex items-center justify-center theme-text-primary hover:opacity-80 transition-colors"
                                    title="Reset zoom"
                                >
                                    <Maximize2 size={14} />
                                </button>
                            </div>
                        )}
                        
                        {/* Zoom level indicator */}
                        {processView.scale !== 1 && (
                            <div className="absolute bottom-4 left-4 z-50 theme-bg-tertiary border theme-border rounded-full px-2 py-1 text-[10px] theme-text-secondary font-bold">
                                {Math.round(processView.scale * 100)}%
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="theme-bg-secondary p-5 rounded-t-3xl border-t theme-border space-y-5 z-20 shrink-0 shadow-[0_-10px_40px_rgba(0,0,0,0.5)] max-h-[40vh] overflow-y-auto">
                {viewMode !== 'original' ? (
                    <>
                        {/* Detection Settings - Always Global */}
                        <div className="flex items-center gap-2 mb-2 justify-between">
                            <div className="flex items-center gap-2">
                                <Settings size={18} className="theme-text-secondary"/>
                                <div className="flex flex-col leading-none">
                                    <span className="font-bold text-sm theme-text-primary">Detection</span>
                                    <span className="text-[10px] theme-text-muted uppercase font-bold mt-0.5">
                                        {segmentMode === 'auto' && 'Auto Contrast'}
                                        {segmentMode === 'manual-bg' && 'Background Ref'}
                                        {segmentMode === 'manual-obj' && 'Object Ref'}
                                    </span>
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <button 
                                    onClick={() => setInvertResult(!invertResult)}
                                    className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 border text-[10px] font-bold uppercase tracking-wider transition-all ${invertResult ? 'bg-purple-600 border-purple-500 text-white shadow-[0_0_10px_rgba(147,51,234,0.3)]' : 'theme-bg-tertiary theme-border theme-text-secondary'}`}
                                >
                                    {invertResult ? <ToggleRight size={14}/> : <ToggleLeft size={14}/>} Invert
                                </button>
                                <div className="flex items-center gap-2 text-xs theme-text-secondary pl-2 border-l theme-border">
                                    <div 
                                        className="w-6 h-6 rounded-full border theme-border shadow-inner" 
                                        style={{backgroundColor: `rgb(${calculatedRefColor.r},${calculatedRefColor.g},${calculatedRefColor.b})`}} 
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-1.5">
                                <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                    <span>Threshold Sensitivity</span>
                                    <span className="theme-text-primary">{threshold}</span>
                                </div>
                                <input 
                                    type="range" min="1" max="150" 
                                    value={threshold} 
                                    onChange={(e) => setThreshold(Number(e.target.value))} 
                                    className="w-full h-1.5 theme-bg-tertiary rounded-lg appearance-none cursor-pointer accent-blue-500" 
                                />
                            </div>
                            
                            <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                        <span>Shadow Removal</span>
                                        <span className="theme-text-primary">{shadowRemoval}</span>
                                    </div>
                                    <input 
                                        type="range" min="0" max="10" 
                                        value={shadowRemoval} 
                                        onChange={(e) => setShadowRemoval(Number(e.target.value))} 
                                        className="w-full h-1.5 theme-bg-tertiary rounded-lg appearance-none cursor-pointer accent-blue-500" 
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                        <span>Detail Scan</span>
                                        <span className="theme-text-primary">{scanStep}px</span>
                                    </div>
                                    <input 
                                        type="range" min="1" max="10" 
                                        value={scanStep} 
                                        onChange={(e) => setScanStep(Number(e.target.value))} 
                                        className="w-full h-1.5 theme-bg-tertiary rounded-lg appearance-none cursor-pointer accent-blue-500" 
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                        <span>Noise Filter</span>
                                        <span className="theme-text-primary">{noiseFilter}px</span>
                                    </div>
                                    <input 
                                        type="range" min="0" max="10" 
                                        value={noiseFilter} 
                                        onChange={(e) => setNoiseFilter(Number(e.target.value))} 
                                        className="w-full h-1.5 theme-bg-tertiary rounded-lg appearance-none cursor-pointer accent-blue-500" 
                                    />
                                </div>
                            </div>
                        </div>
                        
                        {/* Shape Settings - Per-shape when polygons detected (collapsible advanced menu) */}
                        {detectedPolygons.length > 0 && (
                            <div className="mt-4 pt-4 border-t theme-border">
                                <button 
                                    onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                                    className="w-full flex items-center justify-between py-2 px-1 rounded-lg hover:theme-bg-tertiary transition-all"
                                >
                                    <div className="flex items-center gap-2">
                                        <Layers size={16} className="text-emerald-400"/>
                                        <span className="font-bold text-sm theme-text-primary">Shape {selectedPolygonIndex + 1} Settings</span>
                                        <span className="text-[9px] theme-text-muted uppercase tracking-wider">(Advanced)</span>
                                    </div>
                                    {showAdvancedSettings ? <ChevronUp size={16} className="theme-text-secondary"/> : <ChevronDown size={16} className="theme-text-secondary"/>}
                                </button>
                                
                                {showAdvancedSettings && (
                                <>
                                <div className="flex items-center justify-end mt-2 mb-3">
                                    <button 
                                        onClick={resetPolygonToDefaults}
                                        className="px-2 py-1 rounded-lg flex items-center gap-1 theme-bg-tertiary hover:opacity-80 border theme-border theme-text-secondary text-[10px] font-bold uppercase tracking-wider transition-all"
                                        title="Reset to defaults"
                                    >
                                        <RotateCcw size={10}/> Reset
                                    </button>
                                </div>
                                
                                {/* Per-shape Detection Settings */}
                                <div className="space-y-3 mb-4">
                                    <div className="space-y-1.5">
                                        <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                            <span>Threshold</span>
                                            <span className="text-emerald-400">{getSelectedPolygonSettings().threshold}</span>
                                        </div>
                                        <input 
                                            type="range" min="1" max="150" 
                                            value={getSelectedPolygonSettings().threshold} 
                                            onChange={(e) => {
                                                const value = Number(e.target.value);
                                                setDetectedPolygons(prev => {
                                                    const updated = [...prev];
                                                    if (updated[selectedPolygonIndex]) {
                                                        updated[selectedPolygonIndex] = {
                                                            ...updated[selectedPolygonIndex],
                                                            settings: { ...updated[selectedPolygonIndex].settings, threshold: value },
                                                            needsDetectionReprocess: true
                                                        };
                                                    }
                                                    return updated;
                                                });
                                            }} 
                                            className="w-full h-1.5 theme-bg-tertiary rounded-lg appearance-none cursor-pointer accent-emerald-500" 
                                        />
                                    </div>
                                    
                                    <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                                        <div className="space-y-1.5">
                                            <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                                <span>Shadow</span>
                                                <span className="text-emerald-400">{getSelectedPolygonSettings().shadowRemoval}</span>
                                            </div>
                                            <input 
                                                type="range" min="0" max="10" 
                                                value={getSelectedPolygonSettings().shadowRemoval} 
                                                onChange={(e) => {
                                                    const value = Number(e.target.value);
                                                    setDetectedPolygons(prev => {
                                                        const updated = [...prev];
                                                        if (updated[selectedPolygonIndex]) {
                                                            updated[selectedPolygonIndex] = {
                                                                ...updated[selectedPolygonIndex],
                                                                settings: { ...updated[selectedPolygonIndex].settings, shadowRemoval: value },
                                                                needsDetectionReprocess: true
                                                            };
                                                        }
                                                        return updated;
                                                    });
                                                }} 
                                                className="w-full h-1.5 theme-bg-tertiary rounded-lg appearance-none cursor-pointer accent-emerald-500" 
                                            />
                                        </div>
                                        <div className="space-y-1.5">
                                            <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                                <span>Noise</span>
                                                <span className="text-emerald-400">{getSelectedPolygonSettings().noiseFilter}px</span>
                                            </div>
                                            <input 
                                                type="range" min="0" max="10" 
                                                value={getSelectedPolygonSettings().noiseFilter} 
                                                onChange={(e) => {
                                                    const value = Number(e.target.value);
                                                    setDetectedPolygons(prev => {
                                                        const updated = [...prev];
                                                        if (updated[selectedPolygonIndex]) {
                                                            updated[selectedPolygonIndex] = {
                                                                ...updated[selectedPolygonIndex],
                                                                settings: { ...updated[selectedPolygonIndex].settings, noiseFilter: value },
                                                                needsDetectionReprocess: true
                                                            };
                                                        }
                                                        return updated;
                                                    });
                                                }} 
                                                className="w-full h-1.5 theme-bg-tertiary rounded-lg appearance-none cursor-pointer accent-emerald-500" 
                                            />
                                        </div>
                                        <div className="space-y-1.5">
                                            <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                                <span>Scan Step</span>
                                                <span className="text-emerald-400">{getSelectedPolygonSettings().scanStep}px</span>
                                            </div>
                                            <input 
                                                type="range" min="1" max="10" 
                                                value={getSelectedPolygonSettings().scanStep} 
                                                onChange={(e) => {
                                                    const value = Number(e.target.value);
                                                    setDetectedPolygons(prev => {
                                                        const updated = [...prev];
                                                        if (updated[selectedPolygonIndex]) {
                                                            updated[selectedPolygonIndex] = {
                                                                ...updated[selectedPolygonIndex],
                                                                settings: { ...updated[selectedPolygonIndex].settings, scanStep: value },
                                                                needsDetectionReprocess: true
                                                            };
                                                        }
                                                        return updated;
                                                    });
                                                }} 
                                                className="w-full h-1.5 theme-bg-tertiary rounded-lg appearance-none cursor-pointer accent-emerald-500" 
                                            />
                                        </div>
                                    </div>
                                </div>
                                
                                {/* Per-shape Vector Settings */}
                                <div className="grid grid-cols-2 gap-x-6 gap-y-4 pt-3 border-t theme-border-secondary">
                                    <div className="space-y-1.5">
                                        <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                            <span>Curve Smooth</span>
                                            <span className="text-emerald-400">{getSelectedPolygonSettings().curveSmoothing}</span>
                                        </div>
                                        <input 
                                            type="range" min="0" max="5" 
                                            value={getSelectedPolygonSettings().curveSmoothing} 
                                            onChange={(e) => {
                                                const value = Number(e.target.value);
                                                setDetectedPolygons(prev => {
                                                    const updated = [...prev];
                                                    if (updated[selectedPolygonIndex]) {
                                                        updated[selectedPolygonIndex] = {
                                                            ...updated[selectedPolygonIndex],
                                                            settings: { ...updated[selectedPolygonIndex].settings, curveSmoothing: value },
                                                            needsReprocess: true
                                                        };
                                                    }
                                                    return updated;
                                                });
                                            }} 
                                            className="w-full h-1.5 theme-bg-tertiary rounded-lg appearance-none cursor-pointer accent-emerald-500" 
                                        />
                                    </div>
                                    <div className="space-y-1.5">
                                        <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                            <span>Smart Fit</span>
                                            <span className="text-emerald-400">{getSelectedPolygonSettings().smartRefine ? 'ON' : 'OFF'}</span>
                                        </div>
                                        <button 
                                            onClick={() => {
                                                const newValue = !getSelectedPolygonSettings().smartRefine;
                                                setDetectedPolygons(prev => {
                                                    const updated = [...prev];
                                                    if (updated[selectedPolygonIndex]) {
                                                        updated[selectedPolygonIndex] = {
                                                            ...updated[selectedPolygonIndex],
                                                            settings: { ...updated[selectedPolygonIndex].settings, smartRefine: newValue },
                                                            needsReprocess: true
                                                        };
                                                    }
                                                    return updated;
                                                });
                                            }}
                                            className={`w-full py-1.5 rounded-lg flex items-center justify-center gap-1.5 border text-[10px] font-bold uppercase tracking-wider transition-all ${getSelectedPolygonSettings().smartRefine ? 'bg-[var(--accent-emerald)] border-[var(--accent-emerald)] text-white' : 'theme-bg-tertiary theme-border theme-text-secondary'}`}
                                        >
                                            <PenTool size={12}/> {getSelectedPolygonSettings().smartRefine ? 'Enabled' : 'Disabled'}
                                        </button>
                                    </div>
                                    <div className="space-y-1.5">
                                        <div className="flex justify-between text-[10px] uppercase font-bold theme-text-secondary tracking-wider">
                                            <span>Show Holes</span>
                                            <span className="text-amber-400">{getSelectedPolygonSettings().showHoles !== false ? 'ON' : 'OFF'}</span>
                                        </div>
                                        <button 
                                            onClick={() => {
                                                const newValue = getSelectedPolygonSettings().showHoles === false ? true : false;
                                                setDetectedPolygons(prev => {
                                                    const updated = [...prev];
                                                    if (updated[selectedPolygonIndex]) {
                                                        updated[selectedPolygonIndex] = {
                                                            ...updated[selectedPolygonIndex],
                                                            settings: { ...updated[selectedPolygonIndex].settings, showHoles: newValue }
                                                        };
                                                    }
                                                    return updated;
                                                });
                                            }}
                                            className={`w-full py-1.5 rounded-lg flex items-center justify-center gap-1.5 border text-[10px] font-bold uppercase tracking-wider transition-all ${getSelectedPolygonSettings().showHoles !== false ? 'bg-[var(--accent-amber)] border-[var(--accent-amber)] text-black' : 'theme-bg-tertiary theme-border theme-text-secondary'}`}
                                        >
                                            <Circle size={12}/> {getSelectedPolygonSettings().showHoles !== false ? 'Visible' : 'Hidden'}
                                        </button>
                                    </div>
                                </div>
                                </>
                                )}
                            </div>
                        )}

                        <div className="flex gap-2 mt-2 pb-6">
                            <button onClick={() => openSaveDialog('image')} className="flex-1 theme-bg-tertiary hover:opacity-80 border theme-border theme-text-primary font-bold py-3 rounded-xl flex items-center justify-center gap-1.5 text-[10px] transition-all active:scale-[0.98]">
                                <ImageIcon size={14} /> PNG
                            </button>
                            <button onClick={() => openSaveDialog('svg')} disabled={processedPath.length < 3} className="flex-1 bg-[var(--accent-blue)] hover:bg-[var(--accent-blue-hover)] disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-3 rounded-xl flex items-center justify-center gap-1.5 text-[10px] transition-all active:scale-[0.98]">
                                <FileText size={14} /> SVG
                            </button>
                            <button onClick={() => openSaveDialog('dxf')} disabled={processedPath.length < 3} className="flex-1 bg-[var(--accent-emerald)] hover:bg-[var(--accent-emerald-hover)] disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-3 rounded-xl flex items-center justify-center gap-1.5 text-[10px] transition-all active:scale-[0.98] shadow-lg shadow-emerald-900/20">
                                <Download size={14} /> DXF
                            </button>
                        </div>
                    </>
                ) : (
                    <div className="text-center py-8 theme-text-muted text-sm flex flex-col gap-4 items-center pb-12">
                        <p className="max-w-[200px]">Switch to <b>Processed</b> or <b>Heatmap</b> mode above to configure detection settings.</p>
                        <button onClick={() => openSaveDialog('image')} className="w-full max-w-xs theme-bg-tertiary hover:opacity-80 border theme-border theme-text-primary font-bold py-3.5 rounded-xl flex items-center justify-center gap-2 text-xs transition-all">
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