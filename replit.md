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
- Multi-polygon detection (all shapes detected as independent polygons)
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
- DXF export with correct winding direction (CCW for outer contours)
- Polygon selector UI for switching between detected shapes
- Separate layers in DXF for each shape

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
- **Fixed per-polygon bbox preservation bug**: Each polygon now maintains correct pixelBbox
  - Fixed bug where all polygons shared the same pixelBbox after re-detection
  - Now always uses freshly calculated pixelBbox instead of preserving old one
- **UI improvements**:
  - Per-polygon settings now in collapsible "Advanced" menu to reduce clutter
  - Added "Hotwave studio" branding to splash screen
- **Tutorial system for first-time users**:
  - 7-step walkthrough covering all app features
  - Automatically shows for first-time users after splash screen
  - Settings menu (gear icon in header) to toggle tutorial on/off
  - "View Tutorial" button to replay anytime
  - "Don't show again" option to permanently disable
  - Preferences saved to localStorage
- **Dark/Light theme toggle**:
  - Toggle between dark and light interface in Settings menu
  - Theme preference persisted to localStorage
  - Header and main container adapt to theme selection
- **Calibration page system** (complete):
  - Mode selection screen: Quick Scan vs Precision Scan
  - Calibration template generator with corner markers and rulers (SVG)
  - Support for A4, Letter, and Business Card sizes
  - Print/download functionality for calibration templates
  - Corner markers: L-shaped brackets with inner detection patterns
  - **Precision Scan marker detection**:
    - Zhang-Suen skeletonization with 3px border padding
    - Spur pruning to remove anti-aliasing artifacts (<4px branches)
    - Skeleton junction analysis for L-corner detection
    - PCA-based line fitting with perpendicularity validation (5°/10° tolerance)
    - Leg-width consistency check (ratio > 50%)
    - Falls back to edge detection if markers not found
  - **Scan area dimensions**: Uses actual measured distances between L-marker inner corners:
    - Letter: 200mm × 264mm
    - A4: 195mm × 282mm
    - Card: 75.6mm × 44mm
  - **Marker masking**: Corner marker regions are masked during polygon detection in Precision mode
- **Heatmap Guide**: Comprehensive help panel shown in heatmap view mode explaining:
  - What the colors mean (bright = uniform/paper, dark = varied/objects - inverted for reflective objects)
  - How to adjust threshold and use invert
  - Visual legend for color interpretation
  - **Heatmap Contrast slider** (50%-300%) to adjust visibility of detection differences
- **Per-polygon Invert toggle**: Each shape can now have independent invert setting in Advanced settings
  - Useful when individual shapes need opposite detection from global setting
  - Triggers re-detection with per-shape invert value
- **Early polygon initialization**: Polygons are now detected immediately when entering process view
  - Previously only initialized when switching to Contour mode
  - Per-polygon settings now available from the start

### Dec 16, 2024
- **Removed hole detection**: All detected shapes are now treated as independent polygons
  - Internal regions (previously holes) are now detected as separate polygon shapes
  - Simplified DXF/SVG export (no more hole layers)
  - Removed "Show Holes" toggle from UI
- **Fixed per-polygon settings persistence on view mode change**: Settings no longer reset when switching between Original/Heatmap/Contour/Processed views
  - Uses ref tracking to skip polygon re-detection when only view mode changes
- **Increased L-corner detection tolerance**: Precision scan marker detection now accepts up to 30° (first pass) or 45° (fallback) deviation from perpendicular
- **Inverted heatmap visualization**: Uniform areas (paper) now appear bright, varied areas (objects) appear dark - better for reflective/multi-colored objects
