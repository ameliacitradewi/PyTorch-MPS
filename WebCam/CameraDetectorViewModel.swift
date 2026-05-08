import AVFoundation
import Combine
import CoreImage
import CoreML
import CoreVideo
import Foundation
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

#if os(macOS)
import AppKit
#endif

#if os(iOS)
import UIKit
#endif

struct DetectionDisplay: Identifiable, Sendable {
    let id = UUID()
    let label: String
    let confidence: Double
    let boundingBox: CGRect
}

private nonisolated struct PythonFrameRequest: Encodable {
    let frame_id: Int
    let conf: Double
    let jpeg_b64: String
}

private nonisolated struct PythonDetectionPayload: Decodable {
    let label: String
    let confidence: Double
    let x: Double
    let y: Double
    let w: Double
    let h: Double
}

private nonisolated struct PythonFrameResponse: Decodable {
    let ready: Bool?
    let device: String?
    let mps_available: Bool?
    let python_executable: String?
    let torch_version: String?
    let frame_id: Int?
    let detections: [PythonDetectionPayload]?
    let error: String?
}

private nonisolated final class PythonMPSDetectorBridge: @unchecked Sendable {
    var onDetections: (([DetectionDisplay]) -> Void)?
    var onStatus: ((String) -> Void)?
    var onError: ((String) -> Void)?

    private let stateQueue = DispatchQueue(label: "webcam.python.bridge.state")
    private let ciContext = CIContext()
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private let jsonEncoder = JSONEncoder()
    private let jsonDecoder = JSONDecoder()

    private var process: Process?
    private var stdinHandle: FileHandle?
    private var stdoutBuffer = Data()
    private var stderrBuffer = Data()
    private var isReady = false
    private var awaitingResponse = false
    private var lastStderrLine = ""

    init() {}

    func start(
        scriptURL: URL,
        modelURL: URL,
        confidenceThreshold: Double,
        preferredDevice: String
    ) throws {
        var thrownError: Error?

        stateQueue.sync {
            guard process == nil else { return }

            let pythonExecutable: String
            do {
                pythonExecutable = try resolvePythonExecutable()
            } catch {
                thrownError = error
                return
            }

            let process = Process()
            process.executableURL = URL(fileURLWithPath: pythonExecutable)
            process.arguments = [scriptURL.path]

            var environment = ProcessInfo.processInfo.environment
            environment["YOLO_MODEL_PATH"] = modelURL.path
            environment["YOLO_CONF_THRESHOLD"] = String(confidenceThreshold)
            environment["YOLO_TARGET_KEYWORDS"] = "person,human,body,face,head,hand,arm,leg,foot"
            environment["YOLO_CONFIG_DIR"] = FileManager.default.temporaryDirectory
                .appendingPathComponent("ultralytics-config", isDirectory: true).path
            environment["PYTHONUNBUFFERED"] = "1"
            environment["YOLO_DEVICE"] = preferredDevice
            environment["METAL_DEVICE_WRAPPER_TYPE"] = "0"
            environment["MTL_DEBUG_LAYER"] = "0"
            environment["MTL_SHADER_VALIDATION"] = "0"
            environment.removeValue(forKey: "PYTHONPATH")
            process.environment = environment

            let stdinPipe = Pipe()
            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()

            process.standardInput = stdinPipe
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe

            process.terminationHandler = { [weak self] process in
                self?.handleTermination(process)
            }

            do {
                try process.run()
            } catch {
                thrownError = error
                return
            }

            self.process = process
            self.stdinHandle = stdinPipe.fileHandleForWriting
            self.isReady = false
            self.awaitingResponse = false
            self.lastStderrLine = ""
            self.onStatus?("Using Python: \(pythonExecutable)")
            self.onStatus?("Requested device: \(preferredDevice)")

            stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                guard let self else { return }
                let data = handle.availableData
                if !data.isEmpty {
                    self.stateQueue.async {
                        self.consumeStdout(data)
                    }
                }
            }

            stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                guard let self else { return }
                let data = handle.availableData
                if !data.isEmpty {
                    self.stateQueue.async {
                        self.consumeStderr(data)
                    }
                }
            }
        }

        if let thrownError {
            throw thrownError
        }
    }

    func stop() {
        stateQueue.sync {
            if let process, process.isRunning {
                process.terminate()
            }

            stdinHandle = nil
            self.process = nil
            stdoutBuffer.removeAll(keepingCapacity: false)
            stderrBuffer.removeAll(keepingCapacity: false)
            isReady = false
            awaitingResponse = false
            lastStderrLine = ""
        }
    }

    func sendFrame(
        pixelBuffer: CVPixelBuffer,
        frameID: Int,
        confidenceThreshold: Double
    ) -> Bool {
        let canSend = stateQueue.sync { () -> Bool in
            guard let process, process.isRunning else { return false }
            return isReady && !awaitingResponse
        }
        guard canSend else { return false }

        guard let jpegData = encodeJPEG(from: pixelBuffer) else {
            onError?("Failed to encode frame as JPEG.")
            return false
        }

        let request = PythonFrameRequest(
            frame_id: frameID,
            conf: confidenceThreshold,
            jpeg_b64: jpegData.base64EncodedString()
        )

        guard var payload = try? jsonEncoder.encode(request) else {
            onError?("Failed to encode JSON request for Python worker.")
            return false
        }
        payload.append(0x0A)

        var didWrite = false
        stateQueue.sync {
            guard let process, process.isRunning, let stdinHandle else { return }
            do {
                try stdinHandle.write(contentsOf: payload)
                awaitingResponse = true
                didWrite = true
            } catch {
                onError?("Failed to write frame to Python worker: \(error.localizedDescription)")
            }
        }
        return didWrite
    }

    private func encodeJPEG(from pixelBuffer: CVPixelBuffer) -> Data? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return nil
        }

        let jpegData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            jpegData as CFMutableData,
            UTType.jpeg.identifier as CFString,
            1,
            nil
        ) else {
            return nil
        }

        let options: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.72]
        CGImageDestinationAddImage(destination, cgImage, options as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            return nil
        }
        return jpegData as Data
    }

    private func handleTermination(_ process: Process) {
        stateQueue.async { [weak self] in
            guard let self else { return }
            self.stdinHandle = nil
            self.process = nil
            self.isReady = false
            self.awaitingResponse = false
            let detail = self.lastStderrLine.isEmpty ? "" : " Last stderr: \(self.lastStderrLine)"
            self.onError?("Python worker terminated with status \(process.terminationStatus).\(detail)")
        }
    }

    private func consumeStdout(_ data: Data) {
        stdoutBuffer.append(data)
        while let newlineIndex = stdoutBuffer.firstIndex(of: 0x0A) {
            let lineData = stdoutBuffer[..<newlineIndex]
            stdoutBuffer.removeSubrange(...newlineIndex)
            parseStdoutLine(Data(lineData))
        }
    }

    private func consumeStderr(_ data: Data) {
        stderrBuffer.append(data)
        while let newlineIndex = stderrBuffer.firstIndex(of: 0x0A) {
            let lineData = stderrBuffer[..<newlineIndex]
            stderrBuffer.removeSubrange(...newlineIndex)
            if let line = String(data: lineData, encoding: .utf8), !line.isEmpty {
                lastStderrLine = line
                onStatus?("PyTorch: \(line)")
            }
        }
    }

    private func parseStdoutLine(_ data: Data) {
        guard !data.isEmpty else { return }
        guard let response = try? jsonDecoder.decode(PythonFrameResponse.self, from: data) else {
            return
        }

        if response.ready == true {
            isReady = true
            if let device = response.device {
                onStatus?("Runtime device: \(device)")
            }
            if let isMPSAvailable = response.mps_available {
                onStatus?("MPS available: \(isMPSAvailable ? "yes" : "no")")
            }
            if let pythonExecutable = response.python_executable, !pythonExecutable.isEmpty {
                onStatus?("Python runtime: \(pythonExecutable)")
            }
            if let torchVersion = response.torch_version, !torchVersion.isEmpty {
                onStatus?("Torch version: \(torchVersion)")
            }
            onStatus?("PyTorch detector is ready.")
            return
        }

        if let error = response.error, !error.isEmpty {
            awaitingResponse = false
            onError?(error)
            return
        }

        guard let detections = response.detections else { return }
        let mapped = detections.compactMap { payload -> DetectionDisplay? in
            guard payload.confidence.isFinite, payload.confidence > 0 else { return nil }
            let clampedX = min(max(payload.x, 0), 1)
            let clampedY = min(max(payload.y, 0), 1)
            let clampedW = min(max(payload.w, 0), 1)
            let clampedH = min(max(payload.h, 0), 1)
            guard clampedW > 0.001, clampedH > 0.001 else { return nil }

            return DetectionDisplay(
                label: payload.label.capitalized,
                confidence: payload.confidence,
                boundingBox: CGRect(x: clampedX, y: clampedY, width: clampedW, height: clampedH)
            )
        }
        awaitingResponse = false
        onDetections?(mapped)
    }

    private func resolvePythonExecutable() throws -> String {
        let env = ProcessInfo.processInfo.environment
        var candidates: [String] = []

        if let explicit = env["WEBCAM_PYTHON_PATH"], !explicit.isEmpty {
            candidates.append(explicit)
        }

        candidates.append(contentsOf: [
            "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3",
            "/opt/anaconda3/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/bin/python3",
        ])

        var seen: Set<String> = []
        let uniqueCandidates = candidates.filter { seen.insert($0).inserted }

        for path in uniqueCandidates where FileManager.default.isExecutableFile(atPath: path) {
            if pythonHasDependencies(path: path) {
                return path
            }
        }

        throw NSError(
            domain: "PythonMPSDetectorBridge",
            code: 100,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "No compatible Python found. Set WEBCAM_PYTHON_PATH to a Python with torch, ultralytics, opencv-python, and numpy."
            ]
        )
    }

    private func pythonHasDependencies(path: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = ["-c", "import torch, ultralytics, cv2, numpy, PIL"]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        var environment = ProcessInfo.processInfo.environment
        environment["YOLO_CONFIG_DIR"] = FileManager.default.temporaryDirectory
            .appendingPathComponent("ultralytics-config", isDirectory: true).path
        process.environment = environment

        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
}

@MainActor
final class CameraDetectorViewModel: NSObject, ObservableObject {
    nonisolated let session = AVCaptureSession()

    @Published private(set) var detections: [DetectionDisplay] = []
    @Published private(set) var statusMessage = "Preparing camera..."
    @Published private(set) var detectedLabelSummary = ""
    @Published private(set) var personCount = 0
    @Published private(set) var fps: Double = 0
    @Published private(set) var elapsedSeconds: Double = 0
    @Published private(set) var isPaused = false
    @Published private(set) var showScreenshotSavedBanner = false
    @Published private(set) var pausedPreviewImage: CGImage?
    @Published private(set) var screenshotOverlayImage: CGImage?
    @Published private(set) var computeDebugLine = "Compute: unknown"
    @Published private(set) var requestedComputeDebugLine = "Requested Compute: MPS"
    @Published private(set) var pythonDebugLine = "Python: resolving..."
    @Published private(set) var torchDebugLine = "Torch: unknown"

    nonisolated private let sessionQueue = DispatchQueue(label: "webcam.session.queue")
    nonisolated private let inferenceQueue = DispatchQueue(label: "webcam.inference.queue", qos: .userInitiated)
    nonisolated(unsafe) private let videoOutput = AVCaptureVideoDataOutput()
    nonisolated private let ciContext = CIContext()

    nonisolated(unsafe) private var pythonBridge: PythonMPSDetectorBridge?
    nonisolated(unsafe) private var latestPythonDetections: [DetectionDisplay] = []
    nonisolated(unsafe) private var pythonDevicePreference = "mps"
    nonisolated(unsafe) private var didFallbackToCPU = false
    nonisolated(unsafe) private var isRestartingPythonBridge = false

    nonisolated(unsafe) private var model: MLModel?
    nonisolated(unsafe) private var isSessionConfigured = false
    nonisolated private let inferenceIntervalFrames = 5
    nonisolated(unsafe) private var framesUntilNextInference = 0
    nonisolated(unsafe) private var lastDetections: [DetectionDisplay] = []
    nonisolated(unsafe) private var inferenceTimestamps: [CFTimeInterval] = []
    nonisolated(unsafe) private var frameCount = 0
    nonisolated(unsafe) private var inferenceCount = 0
    nonisolated(unsafe) private var totalPersonDetections = 0
    nonisolated(unsafe) private var sessionStartTime = CACurrentMediaTime()
    nonisolated(unsafe) private var pausedForInference = false
    nonisolated(unsafe) private var pendingScreenshot = false
    nonisolated(unsafe) private var pendingPauseFrameCapture = false
    nonisolated(unsafe) private var pausedDisplayFPS: Double = 0
    nonisolated(unsafe) private var pausedDisplayElapsed: Double = 0

    nonisolated private let confidenceThreshold = 0.5

#if os(macOS)
    private var keyMonitor: Any?
#endif

    func start() {
        sessionStartTime = CACurrentMediaTime()
        pythonDevicePreference = initialPythonDevicePreference()
        didFallbackToCPU = false
        isRestartingPythonBridge = false
        framesUntilNextInference = 0
        pausedPreviewImage = nil
        screenshotOverlayImage = nil
        pausedDisplayFPS = 0
        pausedDisplayElapsed = 0
        computeDebugLine = "Compute: initializing..."
        requestedComputeDebugLine = "Requested Compute: \(pythonDevicePreference.uppercased())"
        pythonDebugLine = "Python: resolving..."
        torchDebugLine = "Torch: unknown"
        if pythonDevicePreference == "mps" {
            statusMessage = "Starting YOLO26N .pt worker (prefer MPS)..."
        } else {
            statusMessage = "Starting YOLO26N .pt worker (CPU safe mode)..."
        }
        installKeyboardMonitorIfNeeded()
        requestCameraAccess()
    }

    func stop() {
#if os(macOS)
        removeKeyboardMonitorIfNeeded()
#endif
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.pythonBridge?.stop()
            self.pythonBridge = nil
            self.isRestartingPythonBridge = false
            self.pendingPauseFrameCapture = false
            DispatchQueue.main.async { [weak self] in
                self?.pausedPreviewImage = nil
                self?.screenshotOverlayImage = nil
            }
        }
    }

    func togglePause() {
        let nextState = !isPaused
        isPaused = nextState
        statusMessage = nextState ? "YOLO26N Live Detection (PAUSED)" : "YOLO26N Live Detection"

        if !nextState {
            pausedPreviewImage = nil
        }
        if nextState {
            pausedDisplayFPS = fps
            pausedDisplayElapsed = elapsedSeconds
        }

        inferenceQueue.async { [weak self] in
            guard let self else { return }
            if nextState {
                self.pendingPauseFrameCapture = true
                self.pausedForInference = true
            } else {
                self.pendingPauseFrameCapture = false
                self.pausedForInference = false
            }
        }
    }

    func requestScreenshot() {
        inferenceQueue.async { [weak self] in
            self?.pendingScreenshot = true
        }
    }

    private func requestCameraAccess() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStartSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self else { return }
                    if granted {
                        self.configureAndStartSession()
                    } else {
                        self.statusMessage = "Camera access denied. Please allow camera permission."
                    }
                }
            }
        case .denied, .restricted:
            statusMessage = "Camera access denied. Please allow camera permission."
        @unknown default:
            statusMessage = "Camera permission status is unknown."
        }
    }

    private func configureAndStartSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }

            if !self.isSessionConfigured {
                guard self.configureSession() else { return }
            }

            if !self.session.isRunning {
                self.session.startRunning()
            }
            if !self.session.isRunning {
                self.publishStatus("Camera session failed to start.")
                return
            }

            self.sessionStartTime = CACurrentMediaTime()
            self.inferenceTimestamps.removeAll(keepingCapacity: true)

            do {
                try self.startPythonBridgeIfNeeded()
                self.publishStatus("YOLO26N .pt worker started (device: \(self.pythonDevicePreference)).")
            } catch {
                self.publishStatus("Python worker failed: \(error.localizedDescription)")
            }
        }
    }

    nonisolated private func startPythonBridgeIfNeeded() throws {
        guard pythonBridge == nil else { return }
        guard let scriptURL = resolvePythonScriptURL() else {
            throw NSError(
                domain: "CameraDetectorViewModel",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Cannot find python_mps_detector.py."]
            )
        }

        guard let modelURL = resolveYoloPTURL() else {
            throw NSError(
                domain: "CameraDetectorViewModel",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Cannot find yolo26nbase.pt."]
            )
        }

        let bridge = PythonMPSDetectorBridge()
        bridge.onDetections = { [weak self] detections in
            self?.inferenceQueue.async {
                self?.latestPythonDetections = detections
            }
        }
        bridge.onStatus = { [weak self] message in
            self?.handlePythonStatus(message)
        }
        bridge.onError = { [weak self] message in
            self?.handlePythonBridgeError(message)
        }

        publishRequestedComputeDebug("Requested Compute: \(pythonDevicePreference.uppercased())")

        try bridge.start(
            scriptURL: scriptURL,
            modelURL: modelURL,
            confidenceThreshold: confidenceThreshold,
            preferredDevice: pythonDevicePreference
        )
        pythonBridge = bridge
    }

    nonisolated private func handlePythonBridgeError(_ message: String) {
        let normalized = message.lowercased()
        let mpsCrashDetected = normalized.contains("terminated with status 6")
            || normalized.contains("validatecomputefunctionarguments")

        guard mpsCrashDetected else {
            publishStatus("Python detector error: \(message)")
            return
        }

        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard self.pythonDevicePreference == "mps", !self.didFallbackToCPU else {
                self.publishStatus("Python detector error: \(message)")
                return
            }
            guard !self.isRestartingPythonBridge else { return }
            self.isRestartingPythonBridge = true
            self.didFallbackToCPU = true

            self.publishStatus("MPS backend crashed. Restarting detector on CPU...")
            self.publishComputeDebug("Compute: GPU (MPS) crashed, falling back to CPU")
            self.pythonBridge?.stop()
            self.pythonBridge = nil
            self.pythonDevicePreference = "cpu"
            self.publishRequestedComputeDebug("Requested Compute: CPU (fallback)")

            do {
                try self.startPythonBridgeIfNeeded()
                self.publishStatus("Detector switched to CPU because MPS is unstable.")
            } catch {
                self.publishStatus("CPU fallback failed: \(error.localizedDescription)")
            }
            self.isRestartingPythonBridge = false
        }
    }

    private func initialPythonDevicePreference() -> String {
        let env = ProcessInfo.processInfo.environment
        if let forced = env["WEBCAM_INFERENCE_DEVICE"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           forced == "mps" || forced == "cpu" {
            return forced
        }
        // Default behavior: prefer Apple GPU via MPS.
        return "mps"
    }

    nonisolated private func handlePythonStatus(_ message: String) {
        publishStatus(message)
        let lowered = message.lowercased()

        if message.hasPrefix("Using Python: ") || message.hasPrefix("Python runtime: ") {
            publishPythonDebug(message.replacingOccurrences(of: "Using ", with: ""))
        }
        if message.hasPrefix("Requested device: ") {
            let value = message.replacingOccurrences(of: "Requested device: ", with: "").uppercased()
            publishRequestedComputeDebug("Requested Compute: \(value)")
        }
        if message.hasPrefix("Torch version: ") {
            publishTorchDebug(message)
        }

        if lowered.contains("runtime device: mps") || lowered.contains("pytorch: device: mps") {
            publishComputeDebug("Compute: GPU (MPS)")
        } else if lowered.contains("runtime device: cpu") || lowered.contains("pytorch: device: cpu") {
            publishComputeDebug("Compute: CPU")
        } else if lowered.contains("switched to cpu") || lowered.contains("fallback to cpu") {
            publishComputeDebug("Compute: CPU (fallback)")
        }
    }

    nonisolated private func resolvePythonScriptURL() -> URL? {
        if let bundleURL = Bundle.main.url(forResource: "python_mps_detector", withExtension: "py") {
            return bundleURL
        }

        let sourceURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("python_mps_detector.py")
        if FileManager.default.fileExists(atPath: sourceURL.path) {
            return sourceURL
        }
        return nil
    }

    nonisolated private func resolveYoloPTURL() -> URL? {
        if let bundleURL = Bundle.main.url(forResource: "yolo26nbase", withExtension: "pt") {
            return bundleURL
        }

        let sourceURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("yolo26nbase.pt")
        if FileManager.default.fileExists(atPath: sourceURL.path) {
            return sourceURL
        }
        return nil
    }

    nonisolated private func configureSession() -> Bool {
        guard !isSessionConfigured else { return true }

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        if session.canSetSessionPreset(.vga640x480) {
            session.sessionPreset = .vga640x480
        }

        for input in session.inputs {
            session.removeInput(input)
        }

        let preferredDevice: AVCaptureDevice? = {
#if os(iOS)
            let discovery = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInWideAngleCamera, .builtInDualWideCamera],
                mediaType: .video,
                position: .back
            )
            return discovery.devices.first ?? AVCaptureDevice.default(for: .video)
#else
            let discovery = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.external, .builtInWideAngleCamera],
                mediaType: .video,
                position: .unspecified
            )
            let devices = discovery.devices
            if let external = devices.first(where: { $0.deviceType == .external }) {
                return external
            }
            if let builtIn = devices.first(where: { $0.deviceType == .builtInWideAngleCamera }) {
                return builtIn
            }
            return devices.first ?? AVCaptureDevice.default(for: .video)
#endif
        }()
        guard let cameraDevice = preferredDevice else {
            publishStatus("No camera device found.")
            return false
        }

        do {
            let cameraInput = try AVCaptureDeviceInput(device: cameraDevice)
            guard session.canAddInput(cameraInput) else {
                publishStatus("Unable to attach camera input.")
                return false
            }
            session.addInput(cameraInput)
        } catch {
            publishStatus("Camera input error: \(error.localizedDescription)")
            return false
        }

        do {
            try cameraDevice.lockForConfiguration()
            let supported = cameraDevice.activeFormat.videoSupportedFrameRateRanges.contains { range in
                range.minFrameRate <= 30 && range.maxFrameRate >= 30
            }
            if supported {
                cameraDevice.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
                cameraDevice.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            }
            cameraDevice.unlockForConfiguration()
        } catch {
            publishStatus("Camera frame-rate config warning: \(error.localizedDescription)")
        }

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        videoOutput.setSampleBufferDelegate(self, queue: inferenceQueue)

        guard session.canAddOutput(videoOutput) else {
            publishStatus("Unable to attach camera output.")
            return false
        }
        session.addOutput(videoOutput)

        isSessionConfigured = true
        return true
    }

    nonisolated private func loadModelIfNeeded() throws {
        guard model == nil else { return }
        guard let modelURL = Bundle.main.url(forResource: "yolo26n", withExtension: "mlpackage") else {
            throw NSError(
                domain: "CameraDetectorViewModel",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Cannot find yolo26n.mlpackage in app bundle."]
            )
        }

        let compiledURL = try MLModel.compileModel(at: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: compiledURL, configuration: configuration)
    }

    nonisolated private func publishStatus(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.statusMessage = message
        }
    }

    nonisolated private func publishComputeDebug(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.computeDebugLine = message
        }
    }

    nonisolated private func publishRequestedComputeDebug(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.requestedComputeDebugLine = message
        }
    }

    nonisolated private func publishPythonDebug(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.pythonDebugLine = message
        }
    }

    nonisolated private func publishTorchDebug(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.torchDebugLine = message
        }
    }

    nonisolated private func publishFrameState(
        detections: [DetectionDisplay],
        fps: Double,
        elapsed: Double
    ) {
        let summary = detections
            .prefix(3)
            .map { "\($0.label) \(Int(($0.confidence * 100).rounded()))%" }
            .joined(separator: ", ")

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.detections = detections
            self.personCount = detections.count
            self.fps = fps
            self.elapsedSeconds = elapsed
            self.detectedLabelSummary = summary
        }
    }

    nonisolated private static func rollingFPS(from timestamps: [CFTimeInterval]) -> Double {
        guard timestamps.count >= 2, let first = timestamps.first, let last = timestamps.last, last > first else {
            return 0
        }
        return Double(timestamps.count - 1) / (last - first)
    }

    nonisolated private static func value(in array: MLMultiArray, linearIndex: Int) -> Double {
        switch array.dataType {
        case .double:
            return array.dataPointer.assumingMemoryBound(to: Double.self)[linearIndex]
        case .float32:
            return Double(array.dataPointer.assumingMemoryBound(to: Float.self)[linearIndex])
        case .float16:
            let raw = array.dataPointer.assumingMemoryBound(to: UInt16.self)[linearIndex]
            return float16ToDouble(raw)
        default:
            return array[linearIndex].doubleValue
        }
    }

    nonisolated private static func float16ToDouble(_ value: UInt16) -> Double {
        let sign = UInt32(value & 0x8000) << 16
        var exponent = UInt32(value & 0x7C00) >> 10
        var fraction = UInt32(value & 0x03FF)
        let floatBits: UInt32

        if exponent == 0 {
            if fraction == 0 {
                floatBits = sign
            } else {
                exponent = 1
                while (fraction & 0x0400) == 0 {
                    fraction <<= 1
                    exponent -= 1
                }
                fraction &= 0x03FF
                let adjustedExponent = exponent + (127 - 15)
                floatBits = sign | (adjustedExponent << 23) | (fraction << 13)
            }
        } else if exponent == 0x1F {
            floatBits = sign | 0x7F800000 | (fraction << 13)
        } else {
            let adjustedExponent = exponent + (127 - 15)
            floatBits = sign | (adjustedExponent << 23) | (fraction << 13)
        }

        return Double(Float(bitPattern: floatBits))
    }

    nonisolated private static func normalizedRect(
        rawX: Double,
        rawY: Double,
        rawWOrX2: Double,
        rawHOrY2: Double
    ) -> CGRect? {
        guard rawX.isFinite, rawY.isFinite, rawWOrX2.isFinite, rawHOrY2.isFinite else {
            return nil
        }

        // YOLO CoreML exports may produce either xyxy or xywh.
        var x1 = rawX
        var y1 = rawY
        var x2 = rawWOrX2
        var y2 = rawHOrY2

        if !(rawWOrX2 > rawX && rawHOrY2 > rawY) {
            x1 = rawX - (rawWOrX2 / 2)
            y1 = rawY - (rawHOrY2 / 2)
            x2 = rawX + (rawWOrX2 / 2)
            y2 = rawY + (rawHOrY2 / 2)
        }

        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 2 {
            x1 /= 640
            x2 /= 640
            y1 /= 640
            y2 /= 640
        }

        if x2 < x1 { swap(&x1, &x2) }
        if y2 < y1 { swap(&y1, &y2) }

        x1 = min(max(x1, 0), 1)
        y1 = min(max(y1, 0), 1)
        x2 = min(max(x2, 0), 1)
        y2 = min(max(y2, 0), 1)

        let width = x2 - x1
        let height = y2 - y1
        guard width > 0.001, height > 0.001 else { return nil }
        return CGRect(x: x1, y: y1, width: width, height: height)
    }

    nonisolated private static func extractPersonDetections(
        confidenceArray: MLMultiArray,
        coordinatesArray: MLMultiArray,
        confidenceThreshold: Double
    ) -> [DetectionDisplay] {
        let confidenceShape = confidenceArray.shape.map(\.intValue)
        let coordinatesShape = coordinatesArray.shape.map(\.intValue)

        guard confidenceShape.count == 2, coordinatesShape.count == 2 else { return [] }
        guard confidenceShape[0] > 0, confidenceShape[1] > 0 else { return [] }
        guard coordinatesShape[0] == confidenceShape[0], coordinatesShape[1] >= 4 else { return [] }

        let rowCount = confidenceShape[0]
        let confidenceStrides = confidenceArray.strides.map(\.intValue)
        let coordinateStrides = coordinatesArray.strides.map(\.intValue)

        var personDetections: [DetectionDisplay] = []
        personDetections.reserveCapacity(rowCount)

        for row in 0..<rowCount {
            let personConfidence = value(in: confidenceArray, linearIndex: row * confidenceStrides[0])
            guard personConfidence >= confidenceThreshold else { continue }

            let x = value(in: coordinatesArray, linearIndex: row * coordinateStrides[0])
            let y = value(in: coordinatesArray, linearIndex: row * coordinateStrides[0] + coordinateStrides[1])
            let wOrX2 = value(in: coordinatesArray, linearIndex: row * coordinateStrides[0] + coordinateStrides[1] * 2)
            let hOrY2 = value(in: coordinatesArray, linearIndex: row * coordinateStrides[0] + coordinateStrides[1] * 3)

            guard let rect = normalizedRect(rawX: x, rawY: y, rawWOrX2: wOrX2, rawHOrY2: hOrY2) else {
                continue
            }

            personDetections.append(
                DetectionDisplay(
                    label: "Person",
                    confidence: personConfidence,
                    boundingBox: rect
                )
            )
        }

        return personDetections
    }

    nonisolated private func saveScreenshot(pixelBuffer: CVPixelBuffer, frameNumber: Int) {
#if os(macOS)
        if pausedForInference {
            Task { @MainActor [weak self] in
                self?.saveWindowScreenshot(frameNumber: frameNumber)
            }
            return
        }

        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            publishStatus("Screenshot failed: unable to convert frame.")
            return
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.screenshotOverlayImage = cgImage
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.saveWindowScreenshot(frameNumber: frameNumber)
                self.screenshotOverlayImage = nil
            }
        }
#elseif os(iOS)
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            publishStatus("Screenshot failed: unable to convert frame.")
            return
        }

        do {
            let screenshotFolder = FileManager.default
                .urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("screenshots", isDirectory: true)
            try FileManager.default.createDirectory(at: screenshotFolder, withIntermediateDirectories: true)
            let timestamp = Self.timestampFormatter.string(from: Date())
            let fileURL = screenshotFolder.appendingPathComponent("1-YOLO26N_\(timestamp)_\(frameNumber).png")
            guard let destination = CGImageDestinationCreateWithURL(
                fileURL as CFURL,
                UTType.png.identifier as CFString,
                1,
                nil
            ) else {
                publishStatus("Screenshot failed: unable to create destination.")
                return
            }
            CGImageDestinationAddImage(destination, cgImage, nil)
            guard CGImageDestinationFinalize(destination) else {
                publishStatus("Screenshot failed: writing png data failed.")
                return
            }

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.showScreenshotSavedBanner = true
                self.statusMessage = "Screenshot saved to \(fileURL.path)"
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                    self?.showScreenshotSavedBanner = false
                }
            }
        } catch {
            publishStatus("Screenshot failed: \(error.localizedDescription)")
        }
#else
        publishStatus("Screenshot is not supported on this platform.")
#endif
    }

#if os(macOS)
    @MainActor
    private func saveWindowScreenshot(frameNumber: Int) {
        guard let window = NSApplication.shared.keyWindow,
              let contentView = window.contentView
        else {
            statusMessage = "Screenshot failed: app window is not active."
            return
        }

        let bounds = contentView.bounds
        guard let bitmapRep = contentView.bitmapImageRepForCachingDisplay(in: bounds) else {
            statusMessage = "Screenshot failed: unable to create bitmap."
            return
        }

        contentView.cacheDisplay(in: bounds, to: bitmapRep)
        guard let pngData = bitmapRep.representation(using: .png, properties: [:]) else {
            statusMessage = "Screenshot failed: png encoding failed."
            return
        }

        do {
            let screenshotFolder = FileManager.default
                .urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("screenshots", isDirectory: true)
            try FileManager.default.createDirectory(at: screenshotFolder, withIntermediateDirectories: true)
            let timestamp = Self.timestampFormatter.string(from: Date())
            let fileURL = screenshotFolder.appendingPathComponent("1-YOLO26N_\(timestamp)_\(frameNumber).png")
            try pngData.write(to: fileURL)

            showScreenshotSavedBanner = true
            statusMessage = "Screenshot saved to \(fileURL.path)"
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                self?.showScreenshotSavedBanner = false
            }
        } catch {
            statusMessage = "Screenshot failed: \(error.localizedDescription)"
        }
    }

    private func installKeyboardMonitorIfNeeded() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            guard let key = event.charactersIgnoringModifiers?.lowercased() else { return event }

            switch key {
            case "p":
                self.togglePause()
                return nil
            case "s":
                self.requestScreenshot()
                return nil
            case "q":
                NSApplication.shared.terminate(nil)
                return nil
            default:
                return event
            }
        }
    }

    private func removeKeyboardMonitorIfNeeded() {
        if let keyMonitor {
            NSEvent.removeMonitor(keyMonitor)
            self.keyMonitor = nil
        }
    }
#endif

    nonisolated private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()
}

extension CameraDetectorViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let now = CACurrentMediaTime()

        if pendingScreenshot {
            pendingScreenshot = false
            saveScreenshot(pixelBuffer: pixelBuffer, frameNumber: frameCount)
        }

        if pendingPauseFrameCapture {
            pendingPauseFrameCapture = false
            let image = CIImage(cvPixelBuffer: pixelBuffer)
            if let cgImage = ciContext.createCGImage(image, from: image.extent) {
                DispatchQueue.main.async { [weak self] in
                    self?.pausedPreviewImage = cgImage
                }
            }
        }

        if pausedForInference {
            publishFrameState(
                detections: lastDetections,
                fps: pausedDisplayFPS,
                elapsed: max(0, now - sessionStartTime)
            )
            return
        }

        frameCount += 1
        let rollingFPS = Self.rollingFPS(from: inferenceTimestamps)
        let elapsed = max(0, now - sessionStartTime)

        if framesUntilNextInference > 0 {
            framesUntilNextInference -= 1
            publishFrameState(detections: lastDetections, fps: rollingFPS, elapsed: elapsed)
            return
        }
        framesUntilNextInference = max(0, inferenceIntervalFrames - 1)

        guard let pythonBridge else {
            publishStatus("Python detector is unavailable.")
            publishFrameState(detections: [], fps: rollingFPS, elapsed: elapsed)
            return
        }

        if pythonBridge.sendFrame(
            pixelBuffer: pixelBuffer,
            frameID: frameCount,
            confidenceThreshold: confidenceThreshold
        ) {
            inferenceCount += 1
            inferenceTimestamps.append(now)
            if inferenceTimestamps.count > 30 {
                inferenceTimestamps.removeFirst(inferenceTimestamps.count - 30)
            }
        }

        lastDetections = latestPythonDetections
        totalPersonDetections += lastDetections.count

        publishFrameState(
            detections: lastDetections,
            fps: Self.rollingFPS(from: inferenceTimestamps),
            elapsed: elapsed
        )
    }
}

#if os(iOS)
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.videoGravity = .resize
        view.previewLayer.session = session
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        uiView.previewLayer.session = session
    }
}

final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}
#elseif os(macOS)
struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession

    func makeNSView(context: Context) -> PreviewNSView {
        let view = PreviewNSView()
        view.previewLayer.videoGravity = .resize
        view.previewLayer.session = session
        return view
    }

    func updateNSView(_ nsView: PreviewNSView, context: Context) {
        nsView.previewLayer.session = session
    }
}

final class PreviewNSView: NSView {
    let previewLayer = AVCaptureVideoPreviewLayer()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer = CALayer()
        previewLayer.frame = bounds
        layer?.addSublayer(previewLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        previewLayer.frame = bounds
    }
}
#endif
