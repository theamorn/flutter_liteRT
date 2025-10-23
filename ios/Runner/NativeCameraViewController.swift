import UIKit
import AVFoundation
import TensorFlowLite
import Accelerate

class NativeCameraViewController: UIViewController {
    
    // MARK: - Properties
    private var previewView: UIView!
    private var resultLabel: UILabel!
    private var confidenceLabel: UILabel!
    private var inferenceTimeLabel: UILabel!
    private var timingDetailLabel: UILabel!
    private var modeLabel: UILabel!
    private var gpuToggleButton: UIButton!
    private var closeButton: UIButton!
    
    private var captureSession: AVCaptureSession?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var videoOutput: AVCaptureVideoDataOutput?
    
    private var interpreter: Interpreter?
    private var labels: [String] = []
    private var useGpu = false
    private var isProcessing = false
    
    private let inferenceQueue = DispatchQueue(label: "inference-queue")
    private let modelKey: String
    private let labelsKey: String
    private let imageSize = 192
    private let numClasses = 2024
    
    // MARK: - Initializer
    init(modelKey: String, labelsKey: String) {
        self.modelKey = modelKey
        self.labelsKey = labelsKey
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadModelAndLabels()
        setupCamera()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession?.stopRunning()
    }
    
    deinit {
        interpreter = nil
    }
    
    // MARK: - UI Setup
    private func setupUI() {
        view.backgroundColor = .black
        
        // Preview View
        previewView = UIView()
        previewView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(previewView)
        
        // Result Label
        resultLabel = UILabel()
        resultLabel.translatesAutoresizingMaskIntoConstraints = false
        resultLabel.textAlignment = .center
        resultLabel.textColor = .white
        resultLabel.font = .systemFont(ofSize: 24, weight: .bold)
        resultLabel.text = "Loading..."
        resultLabel.numberOfLines = 0
        view.addSubview(resultLabel)
        
        // Confidence Label
        confidenceLabel = UILabel()
        confidenceLabel.translatesAutoresizingMaskIntoConstraints = false
        confidenceLabel.textAlignment = .center
        confidenceLabel.textColor = .white
        confidenceLabel.font = .systemFont(ofSize: 18)
        confidenceLabel.text = "Confidence: 0%"
        view.addSubview(confidenceLabel)
        
        // Inference Time Label
        inferenceTimeLabel = UILabel()
        inferenceTimeLabel.translatesAutoresizingMaskIntoConstraints = false
        inferenceTimeLabel.textAlignment = .center
        inferenceTimeLabel.textColor = .white
        inferenceTimeLabel.font = .systemFont(ofSize: 16, weight: .bold)
        inferenceTimeLabel.text = "Total: 0ms"
        view.addSubview(inferenceTimeLabel)
        
        // Timing Detail Label
        timingDetailLabel = UILabel()
        timingDetailLabel.translatesAutoresizingMaskIntoConstraints = false
        timingDetailLabel.textAlignment = .center
        timingDetailLabel.textColor = UIColor.white.withAlphaComponent(0.8)
        timingDetailLabel.font = UIFont.monospacedSystemFont(ofSize: 14, weight: .regular)
        timingDetailLabel.text = ""
        timingDetailLabel.numberOfLines = 0
        view.addSubview(timingDetailLabel)
        
        // Mode Label
        modeLabel = UILabel()
        modeLabel.translatesAutoresizingMaskIntoConstraints = false
        modeLabel.textAlignment = .center
        modeLabel.textColor = .white
        modeLabel.font = .systemFont(ofSize: 16)
        modeLabel.text = "CPU Mode"
        view.addSubview(modeLabel)
        
        // GPU Toggle Button
        gpuToggleButton = UIButton(type: .system)
        gpuToggleButton.translatesAutoresizingMaskIntoConstraints = false
        gpuToggleButton.setTitle("Toggle GPU", for: .normal)
        gpuToggleButton.backgroundColor = UIColor.white.withAlphaComponent(0.3)
        gpuToggleButton.setTitleColor(.white, for: .normal)
        gpuToggleButton.layer.cornerRadius = 8
        gpuToggleButton.addTarget(self, action: #selector(toggleGpu), for: .touchUpInside)
        view.addSubview(gpuToggleButton)
        
        // Close Button
        closeButton = UIButton(type: .system)
        closeButton.translatesAutoresizingMaskIntoConstraints = false
        closeButton.setTitle("✕", for: .normal)
        closeButton.titleLabel?.font = .systemFont(ofSize: 28, weight: .bold)
        closeButton.setTitleColor(.white, for: .normal)
        closeButton.addTarget(self, action: #selector(closeCamera), for: .touchUpInside)
        view.addSubview(closeButton)
        
        // Constraints
        NSLayoutConstraint.activate([
            previewView.topAnchor.constraint(equalTo: view.topAnchor),
            previewView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            previewView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            previewView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            closeButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            closeButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            closeButton.widthAnchor.constraint(equalToConstant: 44),
            closeButton.heightAnchor.constraint(equalToConstant: 44),
            
            resultLabel.bottomAnchor.constraint(equalTo: confidenceLabel.topAnchor, constant: -8),
            resultLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            resultLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            confidenceLabel.bottomAnchor.constraint(equalTo: inferenceTimeLabel.topAnchor, constant: -8),
            confidenceLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            confidenceLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            inferenceTimeLabel.bottomAnchor.constraint(equalTo: timingDetailLabel.topAnchor, constant: -4),
            inferenceTimeLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            inferenceTimeLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            timingDetailLabel.bottomAnchor.constraint(equalTo: modeLabel.topAnchor, constant: -8),
            timingDetailLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            timingDetailLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            modeLabel.bottomAnchor.constraint(equalTo: gpuToggleButton.topAnchor, constant: -16),
            modeLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            modeLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            gpuToggleButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            gpuToggleButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            gpuToggleButton.widthAnchor.constraint(equalToConstant: 150),
            gpuToggleButton.heightAnchor.constraint(equalToConstant: 50),
        ])
    }
    
    // MARK: - Model Loading
    private func loadModelAndLabels() {
        inferenceQueue.async { [weak self] in
            guard let weakSelf = self else { return }
            
            do {
                // Load model from Flutter assets using the lookup key
                guard let modelPath = Bundle.main.path(forResource: weakSelf.modelKey, ofType: nil) else {
                    print("❌ Model file not found with key: \(weakSelf.modelKey)")
                    DispatchQueue.main.async {
                        weakSelf.resultLabel.text = "Model not found"
                    }
                    return
                }
                
                print("✓ Model found at: \(modelPath)")
                
                if weakSelf.useGpu {
                    // Try to use CoreML delegate for GPU acceleration
                    do {
                        let coreMLDelegate = CoreMLDelegate()
                        weakSelf.interpreter = try Interpreter(modelPath: modelPath, delegates: [coreMLDelegate!])
                        print("✓ Using CoreML delegate (GPU)")
                    } catch {
                        print("⚠️ CoreML delegate failed, using CPU: \(error)")
                        weakSelf.interpreter = try Interpreter(modelPath: modelPath)
                    }
                } else {
                    weakSelf.interpreter = try Interpreter(modelPath: modelPath)
                    print("✓ Using CPU")
                }
            
                try weakSelf.interpreter?.allocateTensors()
                
                print("✓ Model loaded successfully")
                if let inputShape = try? weakSelf.interpreter?.input(at: 0).shape {
                    print("  Input shape: \(inputShape)")
                }
                if let outputShape = try? weakSelf.interpreter?.output(at: 0).shape {
                    print("  Output shape: \(outputShape)")
                }
                
                // Load labels
                weakSelf.loadLabels()
                
                DispatchQueue.main.async {
                    weakSelf.updateModeText()
                    weakSelf.resultLabel.text = "Ready"
                }
                
                print("✓ Model and labels ready")
            } catch {
                print("❌ Error loading model: \(error)")
                DispatchQueue.main.async {
                    weakSelf.resultLabel.text = "Error loading model"
                }
            }
        }
    }
    
    private func loadLabels() {
        // Load labels from Flutter assets using the lookup key
        guard let labelsPath = Bundle.main.path(forResource: self.labelsKey, ofType: nil) else {
            print("❌ Labels file not found with key: \(self.labelsKey)")
            labels = (0..<numClasses).map { "Unknown Item \($0)" }
            return
        }
        
        print("✓ Labels found at: \(labelsPath)")
        
        do {
            let labelsData = try String(contentsOfFile: labelsPath, encoding: .utf8)
            let lines = labelsData.components(separatedBy: .newlines)
            
            var labelMap: [Int: String] = [:]
            
            // Parse CSV (skip header)
            for i in 1..<lines.count {
                let line = lines[i].trimmingCharacters(in: .whitespaces)
                if line.isEmpty { continue }
                
                let parts = line.components(separatedBy: ",")
                if parts.count >= 2 {
                    if let id = Int(parts[0]) {
                        labelMap[id] = parts[1]
                    }
                }
            }
            
            // Create labels array
            labels = (0..<numClasses).map { labelMap[$0] ?? "Unknown Item \($0)" }
            
            print("✓ Loaded \(labelMap.count) labels")
        } catch {
            print("❌ Error loading labels: \(error)")
            labels = (0..<numClasses).map { "Unknown Item \($0)" }
        }
    }
    
    private func updateModeText() {
        modeLabel.text = useGpu ? "GPU Mode (Metal)" : "CPU Mode"
    }
    
    // MARK: - Camera Setup
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .high
        
        guard let captureSession = captureSession else { return }
        guard let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ Unable to access back camera")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: backCamera)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
            
            videoOutput = AVCaptureVideoDataOutput()
            videoOutput?.setSampleBufferDelegate(self, queue: inferenceQueue)
            videoOutput?.alwaysDiscardsLateVideoFrames = true
            videoOutput?.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            
            if let videoOutput = videoOutput, captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
            }
            
            // Setup preview layer
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
                self.previewLayer?.videoGravity = .resizeAspectFill
                self.previewLayer?.frame = self.previewView.bounds
                if let previewLayer = self.previewLayer {
                    self.previewView.layer.addSublayer(previewLayer)
                }
                
                DispatchQueue.global(qos: .userInitiated).async {
                    captureSession.startRunning()
                }
            }
        } catch {
            print("❌ Error setting up camera: \(error)")
        }
    }
    
    // MARK: - Image Processing
    // Returns: (data: Data, convertTime: Double, resizeTime: Double)
    private func preprocessImage(_ pixelBuffer: CVPixelBuffer) -> (data: Data, convertTime: Double, resizeTime: Double)? {
        let convertStart = Date()
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        // Convert BGRA pixel buffer to RGB data directly
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        // Calculate center crop to square
        let cropSize = min(width, height)
        let xOffset = (width - cropSize) / 2
        let yOffset = (height - cropSize) / 2
        
        let convertTime = Date().timeIntervalSince(convertStart) * 1000
        
        // Resize and convert
        let resizeStart = Date()
        
        // Output buffer for 192x192 RGB
        let outputSize = imageSize
        let outputCount = outputSize * outputSize * 3
        var outputData = Data(count: outputCount)
        
        outputData.withUnsafeMutableBytes { outputPtr in
            guard let output = outputPtr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            
            for y in 0..<outputSize {
                for x in 0..<outputSize {
                    // Map output coordinates to input coordinates (with center crop)
                    let srcX = xOffset + (x * cropSize) / outputSize
                    let srcY = yOffset + (y * cropSize) / outputSize
                    
                    // BGRA pixel position in source
                    let srcOffset = (srcY * bytesPerRow) + (srcX * 4)
                    
                    // RGB position in output
                    let dstOffset = (y * outputSize + x) * 3
                    
                    // Convert BGRA to RGB (skip A, swap B and R)
                    output[dstOffset + 0] = buffer[srcOffset + 2] // R
                    output[dstOffset + 1] = buffer[srcOffset + 1] // G
                    output[dstOffset + 2] = buffer[srcOffset + 0] // B
                }
            }
        }
        
        let resizeTime = Date().timeIntervalSince(resizeStart) * 1000
        
        return (data: outputData, convertTime: convertTime, resizeTime: resizeTime)
    }
    
    // MARK: - Actions
    @objc private func toggleGpu() {
        isProcessing = true
        useGpu.toggle()
        
        inferenceQueue.async { [weak self] in
            self?.loadModelAndLabels()
            Thread.sleep(forTimeInterval: 0.1)
            self?.isProcessing = false
        }
    }
    
    @objc private func closeCamera() {
        captureSession?.stopRunning()
        dismiss(animated: true)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = previewView.bounds
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension NativeCameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !isProcessing else {
            return
        }
        
        guard let interpreter = interpreter else {
            return
        }
        
        guard labels.count > 0 else {
            return
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        isProcessing = true
        
        let totalStart = Date()
        
        guard let preprocessResult = preprocessImage(pixelBuffer) else {
            isProcessing = false
            return
        }
        
        let inputData = preprocessResult.data
        let convertTime = preprocessResult.convertTime
        let resizeTime = preprocessResult.resizeTime
        
        do {
            // Copy input data to interpreter
            let inferenceStart = Date()
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference
            try interpreter.invoke()
            let inferenceTime = Date().timeIntervalSince(inferenceStart) * 1000
            
            // Get output
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            
            // Convert UINT8 output to float probabilities
            let outputArray = [UInt8](outputData)
            let floatOutput = outputArray.map { Float($0) / 255.0 }
            
            // Find top prediction (skip index 0 - background)
            var maxIndex = -1
            var maxProb: Float = 0
            
            for i in 1..<floatOutput.count {
                if floatOutput[i] > maxProb {
                    maxProb = floatOutput[i]
                    maxIndex = i
                }
            }
            
            let totalTime = Date().timeIntervalSince(totalStart) * 1000
            
            // Update UI
            if maxIndex != -1 && maxIndex < labels.count {
                DispatchQueue.main.async { [weak self] in
                    self?.resultLabel.text = self?.labels[maxIndex] ?? "Unknown"
                    self?.confidenceLabel.text = String(format: "Confidence: %d%%", Int(maxProb * 100))
                    self?.inferenceTimeLabel.text = String(format: "Total: %.0fms", totalTime)
                    self?.timingDetailLabel.text = String(
                        format: "├─ Convert: %.0fms\n├─ Resize: %.0fms\n└─ Inference: %.0fms",
                        convertTime,
                        resizeTime,
                        inferenceTime
                    )
                }
            }
        } catch {
            print("❌ Error during inference: \(error)")
        }
        
        isProcessing = false
    }
}
