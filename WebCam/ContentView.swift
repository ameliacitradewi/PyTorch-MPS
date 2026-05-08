//
//  ContentView.swift
//  WebCam
//
//  Created by Amelia Citra on 07/05/26.
//

import SwiftUI

struct ContentView: View {
#if os(iOS) || os(macOS)
    @StateObject private var detector = CameraDetectorViewModel()
#endif

    var body: some View {
#if os(iOS) || os(macOS)
        ZStack {
            CameraPreview(session: detector.session)
                .ignoresSafeArea()

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
                        .foregroundStyle(.white.opacity(0.9))
                        .lineLimit(2)

                    Text(detector.computeDebugLine)
                        .font(.caption2)
                        .foregroundStyle(.cyan.opacity(0.95))
                        .monospaced()

                    Text(detector.requestedComputeDebugLine)
                        .font(.caption2)
                        .foregroundStyle(.cyan.opacity(0.95))
                        .monospaced()

                    Text(detector.torchDebugLine)
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.85))
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)

                    Text(detector.pythonDebugLine)
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.85))
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)

                    Text("Persons: \(detector.isPaused ? "—" : "\(detector.personCount)")")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundStyle(detector.personCount > 0 && !detector.isPaused ? .red : .gray)

                    Text("FPS: \(String(format: "%.1f", detector.fps))")
                        .font(.title3)
                        .foregroundStyle(.gray)
                        .monospacedDigit()

                    Text("Time: \(String(format: "%.1f", detector.elapsedSeconds))s")
                        .font(.title3)
                        .foregroundStyle(.gray)
                        .monospacedDigit()
                }

                Spacer()

                HStack(alignment: .bottom) {
                    if detector.showScreenshotSavedBanner {
                        Text("SCREENSHOT SAVED!")
                            .font(.title3)
                            .fontWeight(.bold)
                            .foregroundStyle(Color(red: 0.4, green: 1.0, blue: 0.4))
                    }

                    Spacer()

                    VStack(alignment: .leading, spacing: 2) {
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

            if detector.isPaused {
                Text("PAUSED")
                    .font(.system(size: 56, weight: .bold))
                    .foregroundStyle(.red)
            }
        }
        .background(.black)
        .onAppear {
            detector.start()
        }
        .onDisappear {
            detector.stop()
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
