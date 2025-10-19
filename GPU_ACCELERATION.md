# GPU Acceleration Guide

## Overview
This app now supports GPU acceleration for faster TFLite inference using platform-specific delegates.

## How It Works

### Android
- Uses **GPU Delegate V2** for hardware acceleration
- Runs inference on the device's GPU (OpenGL ES 3.1+)
- Falls back to CPU if GPU delegate is not supported

### iOS
- Uses **Metal Delegate** for hardware acceleration
- Leverages Apple's Metal framework
- Falls back to CPU if Metal is not available

## Performance Benefits

### CPU vs GPU Inference
- **CPU**: ~200-300ms per inference (4 threads)
- **GPU**: ~50-100ms per inference (typical)
- **Speed improvement**: 2-4x faster with GPU

### Frame Rate
- Without GPU: ~2-3 FPS inference rate
- With GPU: ~10-15 FPS inference rate possible

## Configuration

GPU acceleration is **enabled by default**. The code automatically:
1. Detects the platform (Android/iOS)
2. Tries to initialize the appropriate GPU delegate
3. Falls back to CPU (4 threads) if GPU is unavailable

### Disable GPU (if needed)
To disable GPU and use only CPU, modify `camera_page.dart`:

```dart
await _liteRTHelper.loadModel(
  modelPath: _selectedModel,
  labelsPath: labelsPath,
  useGpu: false, // Add this parameter
);
```

## Precision Settings

Current setting: `isPrecisionLossAllowed: false` (maximum accuracy)

For even faster inference with slight accuracy trade-off:
```dart
isPrecisionLossAllowed: true, // Faster, but ~1-2% accuracy loss
```

## Requirements

### Android
- OpenGL ES 3.1 or higher
- Android 5.0 (API 21) or higher
- Most modern devices (2016+) support this

### iOS
- iOS 9.0 or higher (Metal support)
- A7 chip or newer (iPhone 5s+, iPad Air+)

## Troubleshooting

### Check if GPU is being used
Look at the console logs when the model loads:
```
✅ GPU delegate enabled (Android)
✅ GPU delegate enabled (iOS Metal)
❌ GPU delegate not available, falling back to CPU
```

### GPU not available?
Possible reasons:
1. Model uses operations not supported by GPU
2. Device doesn't have compatible GPU
3. GPU memory insufficient

The app will automatically use CPU in these cases.

## Model Compatibility

**Important**: Not all TFLite models support GPU acceleration. The model must:
- Use supported operations (Conv2D, DepthwiseConv2D, Add, etc.)
- Have appropriate quantization (GPU prefers float32)
- Fit in GPU memory

Our food classifier model (`food-classifier.tflite`) is compatible with GPU acceleration.

## Performance Tips

1. **Lower resolution**: Already set to `ResolutionPreset.low`
2. **Frame skipping**: Every 3rd frame is captured
3. **Inference interval**: 500ms between inferences
4. **GPU acceleration**: Enabled by default
5. **Multi-threading**: 4 CPU threads as fallback

## Testing Performance

Run the app and observe:
- **Inference Time**: Should show in the blue bar (aim for <100ms with GPU)
- **Camera smoothness**: Should be smooth even during inference
- **Console logs**: Check if GPU delegate was successfully initialized

## Advanced: NNAPI (Android only)

For even better performance on Android, you can try NNAPI:

```dart
var nnApiDelegate = NnApiDelegate();
interpreterOptions.addDelegate(nnApiDelegate);
```

Note: NNAPI support varies by device manufacturer and Android version.
