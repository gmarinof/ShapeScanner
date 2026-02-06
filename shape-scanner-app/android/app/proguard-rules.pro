# Capacitor/WebView - keep JS interface
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# Keep Capacitor plugin classes
-keep class com.getcapacitor.** { *; }
-keep class com.capacitorjs.** { *; }

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Preserve line numbers for debugging stack traces
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Keep annotations
-keepattributes *Annotation*

# AndroidX
-keep class androidx.** { *; }
-keep interface androidx.** { *; }

# WebView
-keepclassmembers class * extends android.webkit.WebViewClient {
    public void *(android.webkit.WebView, java.lang.String, android.graphics.Bitmap);
    public boolean *(android.webkit.WebView, java.lang.String);
    public void *(android.webkit.WebView, java.lang.String);
}
