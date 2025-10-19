import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:camera/camera.dart';

class LiteRTHelper {
  late Interpreter _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  List<String> _labels = [];
  
  final int inputImageSize = 224;
  final int inputChannels = 3;
  
  // Convert raw model output scores to probabilities using softmax
  List<double> _softmax(List<double> scores) {
    double maxScore = scores.reduce(math.max);
    List<double> expScores = scores.map((score) => math.exp(score - maxScore)).toList();
    double sumExp = expScores.reduce((a, b) => a + b);
    return expScores.map((expScore) => expScore / sumExp).toList();
  }
  
  Future<void> loadModel({required String modelPath, String? labelsPath, bool useGpu = true}) async {
    try {
      // Configure interpreter options
      var interpreterOptions = InterpreterOptions();
      
      // Configure acceleration based on mode
      if (useGpu) {
        try {
          if (Platform.isAndroid) {
            // Use REAL GPU delegate for Android
            var gpuDelegate = GpuDelegateV2(
              options: GpuDelegateOptionsV2(
                isPrecisionLossAllowed: true,
                inferencePriority1: 2
              ),
            );
            interpreterOptions.addDelegate(gpuDelegate);
            debugPrint('GPU delegate enabled (Android) - using OpenGL/OpenCL GPU acceleration');
          } else if (Platform.isIOS) {
            // iOS Metal delegate
            var gpuDelegate = GpuDelegate(
              options: GpuDelegateOptions(
                allowPrecisionLoss: true,
              ),
            );
            interpreterOptions.addDelegate(gpuDelegate);
            debugPrint('GPU delegate enabled (iOS Metal)');
          }
        } catch (e) {
          debugPrint('GPU delegate failed, falling back to CPU: $e');
          interpreterOptions.threads = 4;
        }
      } else {
        // CPU mode: use optimal thread count
        interpreterOptions.threads = 4;
        debugPrint('CPU mode with 4 threads');
      }
      
      _interpreter = await Interpreter.fromAsset(modelPath, options: interpreterOptions);
      
      // Create isolate interpreter for async inference
      _isolateInterpreter = await IsolateInterpreter.create(address: _interpreter.address);
      debugPrint('IsolateInterpreter created - inference will run in separate isolate');
      
      final inputTensor = _interpreter.getInputTensor(0);
      debugPrint('Input tensor - shape: ${inputTensor.shape}, type: ${inputTensor.type}');
      
      final outputTensor = _interpreter.getOutputTensor(0);
      final numClasses = outputTensor.shape.last;
      debugPrint('Output tensor - type: ${outputTensor.type}, classes: $numClasses');
      
      // Try to load labels from file if provided
      if (labelsPath != null) {
        try {
          final labelsData = await rootBundle.loadString(labelsPath);
          
          // Check if it's a CSV file
          if (labelsPath.endsWith('.csv')) {
            // Parse CSV format (id,name)
            final lines = labelsData.split('\n');
            final labelMap = <int, String>{};
            
            for (var i = 1; i < lines.length; i++) { // Skip header row
              final line = lines[i].trim();
              if (line.isEmpty) continue;
              
              final parts = line.split(',');
              if (parts.length >= 2) {
                final id = int.tryParse(parts[0]);
                final name = parts[1];
                if (id != null) {
                  labelMap[id] = name;
                }
              }
            }
            
            // Create labels list from map
            _labels = List.generate(numClasses, (index) {
              return labelMap[index] ?? 'Unknown Item $index';
            });
            
            debugPrint('Loaded ${labelMap.length} labels from CSV file');
          } else {
            // Plain text format - one label per line
            _labels = labelsData.split('\n')
                .map((label) => label.trim())
                .where((label) => label.isNotEmpty)
                .toList();
            debugPrint('Loaded ${_labels.length} labels from text file');
          }
        } catch (e) {
          debugPrint('Could not load labels file: $e');
          _labels = List.generate(numClasses, (index) => 'Unknown Item $index');
        }
      } else {
        _labels = List.generate(numClasses, (index) => 'Unknown Item $index');
      }
      
      debugPrint('Model loaded successfully with $numClasses classes');
    } catch (e) {
      debugPrint('Failed to load model or labels: $e');
      rethrow;
    }
  }
  
  // Preprocess image file for model input
  Uint8List _preProcessImage(File imageFile) {
    final img.Image? originalImage = img.decodeImage(imageFile.readAsBytesSync());
    if (originalImage == null) throw Exception("Could not decode image.");
    
    final img.Image resizedImage = img.copyResize(
      originalImage, 
      width: inputImageSize, 
      height: inputImageSize,
      interpolation: img.Interpolation.cubic
    );
    
    final inputBytes = Uint8List(1 * inputImageSize * inputImageSize * inputChannels);
    int pixelIndex = 0;
    
    for (int y = 0; y < inputImageSize; y++) {
      for (int x = 0; x < inputImageSize; x++) {
        final pixel = resizedImage.getPixel(x, y);
        inputBytes[pixelIndex++] = pixel.r.toInt();
        inputBytes[pixelIndex++] = pixel.g.toInt();
        inputBytes[pixelIndex++] = pixel.b.toInt();
      }
    }
    
    return inputBytes;
  }

  // Get top N predictions with confidence scores
  List<Map<String, dynamic>> getTopPredictions(File imageFile, {int topN = 10}) {
    final inputBytes = _preProcessImage(imageFile);
    final input = inputBytes.reshape([1, inputImageSize, inputImageSize, inputChannels]);
    final output = List.filled(_labels.length, 0.0).reshape([1, _labels.length]);
    
    _interpreter.run(input, output);
    
    final List<dynamic> outputList = output[0] as List;
    final List<double> rawOutput = outputList.map((e) => (e as num).toDouble()).toList();
    final List<double> probabilities = _softmax(rawOutput);
    
    // Create predictions list, skipping background class (index 0)
    List<Map<String, dynamic>> predictions = [];
    for (int i = 1; i < probabilities.length; i++) {
      predictions.add({
        'label': _labels[i],
        'confidence': probabilities[i],
        'index': i,
      });
    }
    
    predictions.sort((a, b) => (b['confidence'] as double).compareTo(a['confidence'] as double));
    return predictions.take(topN).toList();
  }
  
  // Real-time camera inference (async using IsolateInterpreter)
  Future<List<dynamic>> predictFromCameraImage(CameraImage image) async {
    final inputBytes = await _preprocessCameraImage(image);
    final input = inputBytes.reshape([1, inputImageSize, inputImageSize, inputChannels]);
    final output = List.filled(_labels.length, 0.0).reshape([1, _labels.length]);
    
    // Use isolate interpreter for non-blocking inference
    if (_isolateInterpreter != null) {
      await _isolateInterpreter!.run(input, output);
    } else {
      // Fallback to synchronous if isolate not available
      _interpreter.run(input, output);
    }
    
    final List<dynamic> outputList = output[0] as List;
    final List<double> rawOutput = outputList.map((e) => (e as num).toDouble()).toList();
    final List<double> probabilities = _softmax(rawOutput);
    
    // Find top prediction, skipping background class (index 0)
    double maxProb = 0;
    int maxIndex = -1;
    for(int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }
    
    return maxIndex != -1 ? [_labels[maxIndex], maxProb] : ['Unknown', 0.0];
  }
  
  // Convert camera image to preprocessed uint8 array
  Future<Uint8List> _preprocessCameraImage(CameraImage image) async {
    final img.Image rgbImage = _convertYUV420ToImage(image);

    // Use compute to move heavy image processing to separate isolate
    final inputBytes = await compute(
      _resizeAndConvertImage,
      _ImageResizeParams(
        image: rgbImage,
        targetSize: inputImageSize,
        channels: inputChannels,
      ),
    );
    
    return inputBytes;
  }

  // Convert YUV420 to RGB
  img.Image _convertYUV420ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    
    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;
    
    final image = img.Image(width: width, height: height);
    
    for (int h = 0; h < height; h++) {
      int uvh = (h / 2).floor();
      
      for (int w = 0; w < width; w++) {
        int uvw = (w / 2).floor();
        
        final yIndex = (h * cameraImage.planes[0].bytesPerRow) + w;
        final uvIndex = (uvh * uvRowStride) + (uvw * uvPixelStride);
        
        final y = cameraImage.planes[0].bytes[yIndex];
        final u = cameraImage.planes[1].bytes[uvIndex];
        final v = cameraImage.planes[2].bytes[uvIndex];
        
        // YUV to RGB conversion with proper coefficients
        int r = (y + v * 1.402 - 179.456).round().clamp(0, 255);
        int g = (y - u * 0.34414 - v * 0.71414 + 135.45984).round().clamp(0, 255);
        int b = (y + u * 1.772 - 226.816).round().clamp(0, 255);
        
        image.setPixelRgba(w, h, r, g, b, 255);
      }
    }
    
    return image;
  }
  
  void close() {
    _isolateInterpreter?.close();
    _interpreter.close();
  }
}

// Helper class for passing data to compute isolate
class _ImageResizeParams {
  final img.Image image;
  final int targetSize;
  final int channels;
  
  _ImageResizeParams({
    required this.image,
    required this.targetSize,
    required this.channels,
  });
}

// Top-level function for compute isolate - resizes image and converts to bytes
Uint8List _resizeAndConvertImage(_ImageResizeParams params) {
  final img.Image resizedImage = img.copyResize(
    params.image,
    width: params.targetSize,
    height: params.targetSize,
    interpolation: img.Interpolation.cubic,
  );
  
  final inputBytes = Uint8List(1 * params.targetSize * params.targetSize * params.channels);
  int pixelIndex = 0;
  
  for (int y = 0; y < params.targetSize; y++) {
    for (int x = 0; x < params.targetSize; x++) {
      final pixel = resizedImage.getPixel(x, y);
      inputBytes[pixelIndex++] = pixel.r.toInt();
      inputBytes[pixelIndex++] = pixel.g.toInt();
      inputBytes[pixelIndex++] = pixel.b.toInt();
    }
  }
  
  return inputBytes;
}