import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'litert_helper.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? _cameraController;
  final LiteRTHelper _liteRTHelper = LiteRTHelper();

  // Use ValueNotifiers for frequently updated values
  final ValueNotifier<String> _result = ValueNotifier<String>('');
  final ValueNotifier<double> _confidence = ValueNotifier<double>(0.0);
  final ValueNotifier<int?> _inferenceTimeMs = ValueNotifier<int?>(null);

  bool _isProcessing = false;
  bool _modelLoaded = false;
  bool _isStreaming = false;
  bool _useGpu = false; // Default to CPU (faster on this device)
  List<CameraDescription>? _cameras;
  Timer? _inferenceTimer;
  CameraImage? _latestImage;
  int _frameCount = 0;

  // Throttle settings - With IsolateInterpreter, we can be more aggressive
  static const int _cpuInferenceIntervalMs =
      300;
  static const int _gpuInferenceIntervalMs =
      100;
  static const int _frameSkip = 2;

  // Fixed model path - only food classifier
  static const String _modelPath = 'assets/food-classifier.tflite';
  static const String _labelsPath = 'assets/aiy_food_V1_labelmap.csv';

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras!.isNotEmpty) {
        _cameraController = CameraController(
          _cameras![0],
          ResolutionPreset.veryHigh,
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup.yuv420,
        );

        await _cameraController!.initialize();

        if (mounted) {
          setState(() {});
        }
      }
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  Future<void> _loadModel() async {
    setState(() {
      _modelLoaded = false;
      _result.value = '';
      _confidence.value = 0.0;
    });

    try {
      await _liteRTHelper.loadModel(
        modelPath: _modelPath,
        labelsPath: _labelsPath,
        useGpu: _useGpu,
      );
      setState(() {
        _modelLoaded = true;
      });
    } catch (e) {
      setState(() {
        _result.value = 'Error loading model: $e';
      });
    }
  }

  void _toggleStreaming() {
    setState(() {
      _isStreaming = !_isStreaming;
    });

    if (_isStreaming) {
      _startStreaming();
    } else {
      _stopStreaming();
    }
  }

  Future<void> _startStreaming() async {
    if (_cameraController == null ||
        !_cameraController!.value.isInitialized ||
        !_modelLoaded) {
      return;
    }

    // Start capturing frames (both modes store frames, never process in stream callback)
    _cameraController!.startImageStream((CameraImage image) {
      if (_isStreaming && !_isProcessing) {
        _frameCount++;
        // Store every Nth frame to reduce memory pressure
        if (_frameCount % _frameSkip == 0) {
          _latestImage = image;
        }
      }
    });

    // Start periodic inference timer - GPU uses shorter interval
    final inferenceInterval =
        _useGpu ? _gpuInferenceIntervalMs : _cpuInferenceIntervalMs;
    _inferenceTimer = Timer.periodic(Duration(milliseconds: inferenceInterval), (
      timer,
    ) async {
      // Skip if already processing or no image available
      if (_isProcessing || _latestImage == null) return;

      _isProcessing = true;
      final imageToProcess = _latestImage;
      _latestImage = null; // Clear to avoid reprocessing same frame

      try {
        final startTime = DateTime.now();

        // Async inference using IsolateInterpreter - runs in separate isolate!
        final prediction = await _liteRTHelper.predictFromCameraImage(
          imageToProcess!,
        );
        final endTime = DateTime.now();
        final inferenceTime = endTime.difference(startTime).inMilliseconds;
        debugPrint(
          'Inference completed in ${inferenceTime}ms (${_useGpu ? "GPU" : "CPU"} mode)',
        );

        if (mounted && _isStreaming) {
          _result.value = prediction[0] as String;
          _confidence.value = prediction[1] as double;
          _inferenceTimeMs.value = inferenceTime;
        }
      } catch (e) {
        debugPrint('Error during inference: $e');
      } finally {
        _isProcessing = false;
      }
    });
  }

  void _stopStreaming() {
    _inferenceTimer?.cancel();
    _inferenceTimer = null;
    _cameraController?.stopImageStream();
    _latestImage = null;
    _frameCount = 0; // Reset frame counter
  }

  @override
  void dispose() {
    _inferenceTimer?.cancel();
    _cameraController?.dispose();
    _liteRTHelper.close();
    _result.dispose();
    _confidence.dispose();
    _inferenceTimeMs.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Food Classifier'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 8.0),
            child: Row(
              children: [
                Icon(
                  Icons.memory,
                  size: 20,
                  color: _useGpu ? Colors.green : Colors.grey,
                ),
                const SizedBox(width: 4),
                Text(
                  _useGpu ? 'GPU' : 'CPU',
                  style: const TextStyle(fontSize: 12),
                ),
                Switch(
                  value: _useGpu,
                  onChanged: (value) {
                    setState(() {
                      _useGpu = value;
                    });
                    // Reload model with new GPU setting
                    if (_modelLoaded) {
                      _stopStreaming();
                      _loadModel();
                    }
                  },
                  activeColor: Colors.green,
                ),
              ],
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // Inference Time Display
          ValueListenableBuilder<int?>(
            valueListenable: _inferenceTimeMs,
            builder: (context, inferenceTime, child) {
              if (inferenceTime == null) return const SizedBox.shrink();

              return Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 12,
                ),
                color: Colors.blue.shade50,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.timer, size: 20, color: Colors.blue.shade700),
                    const SizedBox(width: 8),
                    Text(
                      'Inference Time: $inferenceTime ms',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: Colors.blue.shade700,
                      ),
                    ),
                  ],
                ),
              );
            },
          ),

          // Camera Preview
          Expanded(
            flex: 3,
            child: Container(
              color: Colors.black,
              child:
                  _cameraController != null &&
                          _cameraController!.value.isInitialized
                      ? Stack(
                        fit: StackFit.expand,
                        children: [
                          CameraPreview(_cameraController!),
                          // Overlay with results - using ValueListenableBuilder
                          ValueListenableBuilder<String>(
                            valueListenable: _result,
                            builder: (context, result, child) {
                              if (result.isEmpty)
                                return const SizedBox.shrink();

                              return Positioned(
                                bottom: 0,
                                left: 0,
                                right: 0,
                                child: Container(
                                  padding: const EdgeInsets.all(16),
                                  decoration: BoxDecoration(
                                    gradient: LinearGradient(
                                      begin: Alignment.bottomCenter,
                                      end: Alignment.topCenter,
                                      colors: [
                                        Colors.black.withOpacity(0.8),
                                        Colors.transparent,
                                      ],
                                    ),
                                  ),
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Row(
                                        children: [
                                          const Icon(
                                            Icons.restaurant,
                                            color: Colors.orange,
                                            size: 24,
                                          ),
                                          const SizedBox(width: 8),
                                          Expanded(
                                            child: Text(
                                              result,
                                              style: const TextStyle(
                                                color: Colors.white,
                                                fontSize: 24,
                                                fontWeight: FontWeight.bold,
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                      const SizedBox(height: 8),
                                      ValueListenableBuilder<double>(
                                        valueListenable: _confidence,
                                        builder: (context, confidence, child) {
                                          return Row(
                                            children: [
                                              const Icon(
                                                Icons.analytics,
                                                color: Colors.blue,
                                                size: 20,
                                              ),
                                              const SizedBox(width: 8),
                                              Text(
                                                'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
                                                style: const TextStyle(
                                                  color: Colors.white,
                                                  fontSize: 18,
                                                ),
                                              ),
                                            ],
                                          );
                                        },
                                      ),
                                    ],
                                  ),
                                ),
                              );
                            },
                          ),
                        ],
                      )
                      : const Center(child: CircularProgressIndicator()),
            ),
          ),

          // Controls
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 8,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            child: Column(
              children: [
                if (!_modelLoaded)
                  const Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(width: 16),
                      Text('Loading model...'),
                    ],
                  )
                else
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _toggleStreaming,
                          icon: Icon(
                            _isStreaming ? Icons.stop : Icons.play_arrow,
                          ),
                          label: Text(
                            _isStreaming ? 'Stop Stream' : 'Start Stream',
                          ),
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            backgroundColor:
                                _isStreaming ? Colors.red : Colors.green,
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),
                const SizedBox(height: 12),
                Text(
                  _isStreaming
                      ? 'Camera stream is running...'
                      : 'Press Start to begin real-time classification',
                  style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
