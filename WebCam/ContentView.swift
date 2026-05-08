//
//  ContentView.swift
//  WebCam
//
//  Created by Amelia Citra on 07/05/26.
//

import SwiftUI
import Combine

struct ContentView: View {
#if os(iOS) || os(macOS)
    @StateObject private var detector = CameraDetectorViewModel()
    @State private var currentDate = Date()
    private let clockTimer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "dd-MM-yyyy HH:mm:ss"
        return formatter
    }()
#endif

    var body: some View {
#if os(iOS) || os(macOS)
        ZStack {
            CameraPreview(session: detector.session)
                .ignoresSafeArea()

            if !detector.isPaused, let screenshotImage = detector.screenshotOverlayImage {
                GeometryReader { geometry in
                    Image(decorative: screenshotImage, scale: 1.0)
                        .resizable()
                        .scaledToFill()
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .clipped()
                }
                .ignoresSafeArea()
            }

            if detector.isPaused, let pausedImage = detector.pausedPreviewImage {
                GeometryReader { geometry in
                    Image(decorative: pausedImage, scale: 1.0)
                        .resizable()
                        .scaledToFill()
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .clipped()
                }
                .ignoresSafeArea()
            }

            DetectionOverlay(detections: detector.detections)
                .allowsHitTesting(false)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 0) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("YOLO26N Live Detection")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundStyle(Color(red: 0.4, green: 1.0, blue: 0.4))

                    Text(detector.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(1))
                        .lineLimit(2)

                    Text(detector.computeDebugLine)
                        .font(.caption2)
                        .foregroundStyle(.cyan.opacity(1))
                        .monospaced()

                    Text(detector.requestedComputeDebugLine)
                        .font(.caption2)
                        .foregroundStyle(.cyan.opacity(1))
                        .monospaced()

                    Text(detector.torchDebugLine)
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(1))
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)

                    Text(detector.pythonDebugLine)
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(1))
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)

                }

                Spacer()

                HStack(alignment: .bottom) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Persons: \(detector.isPaused ? "—" : "\(detector.personCount)")")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundStyle(detector.personCount > 0 && !detector.isPaused ? .red : .black)

                        Text("FPS: \(String(format: "%.1f", detector.fps))")
                            .font(.title3)
                            .foregroundStyle(.white)
                            .monospacedDigit()

                        Text("Date: \(Self.dateFormatter.string(from: currentDate))")
                            .font(.title3)
                            .foregroundStyle(.white)
                            .monospacedDigit()

                        Text("Time elapsing: \(String(format: "%.1f", detector.elapsedSeconds))s")
                            .font(.title3)
                            .foregroundStyle(.white)
                            .monospacedDigit()

                        if detector.showScreenshotSavedBanner {
                            Text("SCREENSHOT SAVED!")
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundStyle(Color(red: 0.4, green: 1.0, blue: 0.4))
                        }
                    }

                    Spacer()

                    VStack(alignment: .leading, spacing: 2) {
                        if detector.isPaused {
                            Text("PAUSED")
                                .font(.system(size: 24, weight: .bold))
                                .foregroundStyle(.red)
                                .padding(.bottom, 4)
                        }
                        Text("Press [q]=Quit")
                        Text("[s]=Screenshot")
                        Text("[p]=Pause/Resume")
                    }
                    .font(.title3)
                    .foregroundStyle(Color(red: 0.4, green: 1.0, blue: 0.4))
                    .monospaced()
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .background(.black)
        .onAppear {
            detector.start()
        }
        .onDisappear {
            detector.stop()
        }
        .onReceive(clockTimer) { value in
            currentDate = value
        }
#else
        VStack(spacing: 10) {
            Image(systemName: "camera.metering.unknown")
                .font(.system(size: 48))
            Text("This build supports camera detection on iOS and macOS.")
        }
        .padding()
#endif
    }
}

#if os(iOS) || os(macOS)
private struct DetectionOverlay: View {
    let detections: [DetectionDisplay]

    var body: some View {
        GeometryReader { geometry in
            ForEach(detections) { detection in
                let rect = CGRect(
                    x: detection.boundingBox.minX * geometry.size.width,
                    y: detection.boundingBox.minY * geometry.size.height,
                    width: detection.boundingBox.width * geometry.size.width,
                    height: detection.boundingBox.height * geometry.size.height
                )

                ZStack(alignment: .topLeading) {
                    Rectangle()
                        .stroke(Color(red: 0.0, green: 1.0, blue: 0.0), lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)

                    Text("\(detection.label) \(Int(detection.confidence * 100))%")
                        .font(.title3)
                        .bold()
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color(red: 0.0, green: 1.0, blue: 0.0), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(.black)
                        .offset(x: 2, y: 2)
                }
                .position(
                    x: rect.minX + rect.width / 2,
                    y: rect.minY + rect.height / 2
                )
            }
        }
    }
}
#endif

#Preview {
    ContentView()
}
