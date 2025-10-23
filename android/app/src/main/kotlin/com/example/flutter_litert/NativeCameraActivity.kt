package com.example.flutter_litert

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class NativeCameraActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var resultText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var inferenceTimeText: TextView
    private lateinit var timingDetailText: TextView
    private lateinit var modeText: TextView
    
    @Volatile
    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private lateinit var cameraExecutor: ExecutorService
    private var useGpu = false
    
    @Volatile
    private var isProcessing = false
    
    private val interpreterLock = Any()
    private val inferenceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    companion object {
        private const val TAG = "NativeCameraActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val MODEL_PATH = "flutter_assets/assets/food-classifier.tflite"
        private const val LABELS_PATH = "flutter_assets/assets/aiy_food_V1_labelmap.csv"
        private const val IMAGE_SIZE = 192
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_native_camera)
        
        previewView = findViewById(R.id.previewView)
        resultText = findViewById(R.id.resultText)
        confidenceText = findViewById(R.id.confidenceText)
        inferenceTimeText = findViewById(R.id.inferenceTimeText)
        timingDetailText = findViewById(R.id.timingDetailText)
        modeText = findViewById(R.id.modeText)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        loadModelAndLabels()
        
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
        
        findViewById<View>(R.id.gpuToggle).setOnClickListener {
            isProcessing = true
            useGpu = !useGpu
            
            inferenceScope.launch {
                try {
                    loadModelAndLabels()
                    withContext(Dispatchers.Main) {
                        updateModeText()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error reloading model: ${e.message}", e)
                } finally {
                    kotlinx.coroutines.delay(100)
                    isProcessing = false
                }
            }
        }
    }
    
    private fun loadModelAndLabels() {
        synchronized(interpreterLock) {
            try {
                val compatList = CompatibilityList()
                val options = Interpreter.Options()
                
                if (useGpu && compatList.isDelegateSupportedOnThisDevice) {
                    val gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                } else if (useGpu && !compatList.isDelegateSupportedOnThisDevice) {
                    useGpu = false
                    options.setNumThreads(4)
                } else {
                    options.setNumThreads(4)
                }
                
                interpreter?.close()
                interpreter = null
                interpreter = Interpreter(FileUtil.loadMappedFile(this, MODEL_PATH), options)
                
                labels = loadLabels()
                updateModeText()
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model: ${e.message}", e)
                Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun loadLabels(): List<String> {
        val labelList = mutableListOf<String>()
        try {
            val reader = BufferedReader(InputStreamReader(assets.open(LABELS_PATH)))
            reader.useLines { lines ->
                lines.drop(1) // Skip header
                    .forEach { line ->
                        val parts = line.split(",")
                        if (parts.size >= 2) {
                            labelList.add(parts[1].trim())
                        }
                    }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels: ${e.message}", e)
        }
        return labelList
    }
    
    private fun updateModeText() {
        runOnUiThread {
            modeText.text = if (useGpu) "GPU Mode" else "CPU Mode"
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
            
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        processImage(image)
                    }
                }
            
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    @androidx.camera.core.ExperimentalGetImage
    private fun processImage(imageProxy: ImageProxy) {
        if (isProcessing || interpreter == null) {
            imageProxy.close()
            return
        }
        
        isProcessing = true
        
        inferenceScope.launch {
            try {
                val totalStart = System.currentTimeMillis()
                
                // Convert to bitmap and rotate
                val convertStart = System.currentTimeMillis()
                val bitmap = imageProxy.toBitmap()
                val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees.toFloat())
                val convertTime = System.currentTimeMillis() - convertStart
                
                // Resize and preprocess
                val resizeStart = System.currentTimeMillis()
                val tensorImage = TensorImage(org.tensorflow.lite.DataType.UINT8)
                tensorImage.load(rotatedBitmap)
                val imageProcessor = ImageProcessor.Builder()
                    .add(ResizeWithCropOrPadOp(IMAGE_SIZE, IMAGE_SIZE))
                    .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                    .build()
                
                val processedImage = imageProcessor.process(tensorImage)
                val inputBuffer = processedImage.buffer
                val resizeTime = System.currentTimeMillis() - resizeStart
                
                // Run inference
                val inferenceStart = System.currentTimeMillis()
                val outputArray = Array(1) { ByteArray(labels.size) }
                
                synchronized(interpreterLock) {
                    interpreter?.run(inputBuffer, outputArray)
                }
                val inferenceTime = System.currentTimeMillis() - inferenceStart
                
                val totalTime = System.currentTimeMillis() - totalStart
                
                val floatOutput = FloatArray(labels.size) { i ->
                    (outputArray[0][i].toInt() and 0xFF) / 255.0f
                }
                
                var maxIndex = -1
                var maxProb = 0f
                
                for (i in 1 until floatOutput.size) {
                    if (floatOutput[i] > maxProb) {
                        maxProb = floatOutput[i]
                        maxIndex = i
                    }
                }
                
                if (maxIndex != -1) {
                    withContext(Dispatchers.Main) {
                        resultText.text = labels.getOrNull(maxIndex) ?: "Unknown"
                        confidenceText.text = "Confidence: ${(maxProb * 100).toInt()}%"
                        inferenceTimeText.text = "Total: ${totalTime}ms"
                        timingDetailText.text = "├─ Convert: ${convertTime}ms\n" +
                                "├─ Resize: ${resizeTime}ms\n" +
                                "└─ Inference: ${inferenceTime}ms"
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during inference: ${e.message}", e)
            } finally {
                imageProxy.close()
                isProcessing = false
            }
        }
    }
    
    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        isProcessing = true // Stop processing
        cameraExecutor.shutdown()
        inferenceScope.cancel()
        
        synchronized(interpreterLock) {
            interpreter?.close()
            interpreter = null
        }
    }
}
