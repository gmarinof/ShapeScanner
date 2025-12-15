# ShapeScanner

## Overview
ShapeScanner is a mobile app that scans physical objects from photos and converts them to vectorized DXF files for CAD/CNC manufacturing. Built with React + Vite + Tailwind CSS + Capacitor for Android deployment.

## Project Structure
```
shape-scanner-app/
├── src/
│   ├── App.jsx          # Main application component
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles (Tailwind)
├── android/             # Capacitor Android project
├── public/              # Static assets
├── capacitor.config.json
├── vite.config.js
└── package.json

.github/workflows/
└── build-apk.yml        # GitHub Actions for APK build
```

## Key Features
- Photo capture/upload with perspective correction
- Automatic paper corner detection (Otsu thresholding)
- Multi-polygon detection with hole support
- Per-polygon settings (each shape can have independent slider settings)
- Shape fitting (circles, polygons)
- DXF export for CAD software
- SVG export with mm dimensions
- AI analysis with Gemini Vision (optional)

## Running the App
```bash
cd shape-scanner-app && npm run dev
```

## Building APK
Push to GitHub and trigger the Actions workflow:
```bash
git add .
git commit -m "message"
git push github main
```

## Recent Changes (Dec 2024)
- Multi-polygon detection with 4-connectivity boundary tracing
- Hole detection for shapes like washers and frames with proper component ownership
- DXF export with correct winding direction (CCW for outer contours, CW for holes)
- Polygon selector UI for switching between detected shapes
- Separate layers in DXF for each shape and hole

### Dec 14, 2024
- Camera capture using Capacitor Camera.getPhoto() for native platforms
- Process view zoom/pan: scroll wheel to zoom, middle-click/Alt+drag to pan
- Draggable detection badge (green shape count label)
- Fixed DXF export to include proper unit headers ($INSUNITS=4 for mm)
- Added SVG export option with proper mm dimensions
- **Per-polygon settings**: Each detected shape now has independent settings
  - Curve smoothing and smart refine work per-polygon
  - Other settings (threshold, noise, shadow, invert) are global
  - Green indicator bar shows which shape's settings are being edited
  - Reset button to restore shape to default settings

### Dec 15, 2024
- **Tap-to-select polygons**: Click/tap directly on shapes in the canvas to select them
- **Full per-polygon settings**: ALL settings now work per-shape
  - Detection settings (threshold, shadow, noise, scan, invert) - triggers re-detection via `reprocessPolygonDetection`
  - Vector settings (curve smooth, smart fit) - triggers vector refinement via `reprocessPolygon`
  - Each shape can have completely independent settings
  - "Shape X Settings" section shows all sliders for the selected shape
  - Reset button restores selected shape to global defaults (triggers re-detection)
  - Uses `needsDetectionReprocess` flag for detection settings changes
  - Uses `needsReprocess` flag for vector-only settings changes
  - `unwarpedBufferRef` stores the grayscale buffer for per-polygon ROI reprocessing
- **Fixed per-polygon settings persistence**: Settings now properly persist when switching between polygons
  - Removed `selectedPolygonIndex` from processImage dependencies to prevent full re-detection on polygon switch
  - Added separate lightweight effect for polygon selection display updates
  - Each polygon maintains independent settings even after switching away and back
- **Fixed per-polygon reprocessing coordinates**: Per-polygon detection now correctly updates the actual shape
  - Fixed double-scaling bug in `reprocessPolygonDetection` - now uses full paper dimensions
  - Proper ROI-local to global mm coordinate conversion
  - pixelBbox is recalculated after each reprocess for subsequent edits
- **Improved corner detection with Canny edge detection**:
  - New `EdgeDetector` class with Sobel gradient computation
  - Gaussian blur with mirror boundary handling
  - Non-maximum suppression for edge thinning
  - Canny-style hysteresis thresholding (multiple sensitivity levels)
  - Moore boundary contour tracing (follows edges in order)
  - Douglas-Peucker contour simplification
  - Quadrilateral fitting with convexity and area validation
  - Contrast adjustment now applies to detection (not just display)
  - Falls back to color distance + Otsu thresholding if edge detection fails
  - Extended contrast slider range (25%-300%) for better visibility
  - Paper color picker - tap "Pick" then tap on the paper to select its color
