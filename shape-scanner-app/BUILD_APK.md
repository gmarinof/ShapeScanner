# How to Build the ShapeScanner APK

## Prerequisites
- Android Studio (download from https://developer.android.com/studio)
- Java JDK 17 or higher

## Steps to Build

### 1. Download this project
Download all files from Replit to your local computer.

### 2. Open in Android Studio
```bash
cd shape-scanner-app
npx cap open android
```
Or manually open Android Studio and select: File > Open > navigate to `shape-scanner-app/android`

### 3. Wait for Gradle Sync
Android Studio will automatically sync Gradle. Wait for it to complete (bottom progress bar).

### 4. Build the APK

**For Debug APK (testing):**
- Go to: Build > Build Bundle(s) / APK(s) > Build APK(s)
- APK location: `android/app/build/outputs/apk/debug/app-debug.apk`

**For Release APK (production):**
- Go to: Build > Generate Signed Bundle / APK
- Select APK > Next
- Create a new keystore or use existing
- Select "release" build variant
- Click Create

### 5. Install on Android Device
- Transfer the APK to your phone
- Enable "Install from unknown sources" in settings
- Open the APK file to install

## Making Changes

After modifying the React code:
```bash
npm run build
npx cap sync android
```

Then rebuild in Android Studio.

## App Permissions

The app requests:
- Camera access (for taking photos)
- Storage access (for saving files)

These are configured in `android/app/src/main/AndroidManifest.xml`
