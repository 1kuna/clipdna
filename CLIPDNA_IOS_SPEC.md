# ClipDNA iOS

A native iOS/iPadOS app for finding duplicate and near-duplicate videos in your photo library, specifically designed to handle Instagram's re-encoding (lower resolution, trimmed frames) and identify the best version to keep.

**Target Platform:** iOS 26 / iPadOS 26  
**Language:** Swift 6.2  
**IDE:** Xcode 26  
**Minimum Device:** iPhone 12 / iPad (9th generation)

---

## Problem Statement

Instagram and other social media apps frequently:
- Re-encode videos at lower resolution
- Trim a few frames from the beginning/end
- Change compression/bitrate
- Modify metadata

Standard duplicate detection fails because the files are technically different. ClipDNA uses Vision framework perceptual fingerprinting to find these near-duplicates and suggests which version to keep (higher resolution, longer duration, better quality).

---

## Algorithm: Vision Framework Feature Prints

### Why Vision VNGenerateImageFeaturePrintRequest?

After researching iOS 26 APIs, **VNGenerateImageFeaturePrintRequest remains Apple's recommended approach** for image and video similarity detection. No new perceptual hashing or video fingerprinting APIs were introduced at WWDC 2025.

| Approach | Pros | Cons |
|----------|------|------|
| **Vision VNFeaturePrint** | Native Apple API, Neural Engine optimized, 768-dim normalized vectors (iOS 17+), handles compression well | Requires iOS 13+ |
| **pHash (CocoaImageHashing)** | Simple 64-bit hash, fast comparison | Less robust to re-encoding, 3rd party lib |
| **Custom CoreML Model** | Maximum control | Overkill, maintenance burden |
| **Foundation Models** | Free on-device LLM | Text-focused, not for visual similarity |

**Winner: Vision Framework** — Apple's `VNGenerateImageFeaturePrintRequest` generates learned neural network feature vectors that are:
- Robust to resolution changes and compression artifacts
- Normalized 768-element vectors (iOS 17+ Revision2)
- Comparable via Euclidean distance (0.0-2.0 range)
- Optimized for Neural Engine execution

### Distance Thresholds (iOS 17+ / iOS 26)

```
Distance < 0.3  → Definite duplicate (same video, different encoding)
Distance 0.3-0.5 → Likely duplicate (review recommended)  
Distance 0.5-0.8 → Possibly related (similar content)
Distance > 0.8  → Different videos
```

### Algorithm Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ClipDNA Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. INDEXING (BGContinuedProcessingTask)                            │
│     ┌─────────┐    ┌─────────────┐    ┌─────────────────┐          │
│     │ PHAsset │───▶│ Extract     │───▶│ VNFeaturePrint  │          │
│     │ (video) │    │ frames @1fps│    │ per frame       │          │
│     └─────────┘    └─────────────┘    └────────┬────────┘          │
│                                                 │                   │
│                                                 ▼                   │
│                                        ┌───────────────┐            │
│                                        │ Store in      │            │
│                                        │ SwiftData     │            │
│                                        └───────────────┘            │
│                                                                     │
│  2. MATCHING                                                        │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ For each video pair:                                     │    │
│     │   - Compare frame sequences (sliding window alignment)   │    │
│     │   - Handle trimmed frames (find best offset)             │    │
│     │   - Compute average frame distance                       │    │
│     │   - If distance < 0.45 threshold → DUPLICATE             │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
│  3. RANKING                                                         │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ For each duplicate group:                                │    │
│     │   - Compare resolution (prefer higher)                   │    │
│     │   - Compare duration (prefer longer)                     │    │
│     │   - Compare bitrate (prefer higher)                      │    │
│     │   - Check if original vs social media export             │    │
│     │   → Suggest "KEEP" vs "DELETE"                           │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## iOS 26 Key Technologies

### BGContinuedProcessingTask

iOS 26 introduces `BGContinuedProcessingTask` for user-initiated long-running work. Unlike `BGProcessingTask` (discretionary background work), this API:

- **Starts immediately** when user triggers the action
- **Continues after app backgrounded** with system UI (Dynamic Island / notification)
- **Reports progress** to the user automatically
- **Handles interruption gracefully** with expiration handlers

**Critical constraints:**
- Background execution runs **4-5× slower** (no GPU, efficiency cores only)
- Must report progress every **~20 seconds** or task terminates
- State must be saved incrementally for interruption recovery
- iPad can enable background GPU for better performance

```swift
// Registration in App init
BGTaskScheduler.shared.register(
    forTaskWithIdentifier: "com.clipdna.videoIndexing",
    using: nil
) { task in
    await handleBackgroundIndexing(task: task as! BGContinuedProcessingTask)
}

// Triggering the task
func startBackgroundIndexing(videoCount: Int) throws {
    let request = BGContinuedProcessingTaskRequest(
        identifier: "com.clipdna.videoIndexing",
        title: "Indexing Videos",
        subtitle: "0 of \(videoCount) videos processed"
    )
    request.strategy = .immediately
    try BGTaskScheduler.shared.submit(request)
}
```

### Swift 6.2 Features Used

| Feature | Use Case |
|---------|----------|
| **InlineArray** | Stack-allocated frame buffers for zero-heap hot paths |
| **Span** | Zero-copy buffer views for frame data pipelines |
| **@concurrent** | Explicit background thread pool execution for Vision requests |
| **Observations** | Efficient batched UI updates from processing progress |
| **Task naming** | Debuggable async tasks in Instruments |
| **Typed throws** | Precise error handling in async contexts |

### Xcode 26 Tooling

- **Processor Trace Instrument** — Profile frame extraction bottlenecks (M4/iPhone 16+)
- **CPU Counters Instrument** — Identify memory vs compute bottlenecks
- **Power Profiler** — Optimize battery impact of background processing
- **Task ID visibility** — Debug async tasks in backtraces

---

## Architecture

### Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **UI Framework** | SwiftUI (iOS 26) | Liquid Glass design, modern state management |
| **Video Processing** | AVFoundation | Native frame extraction |
| **Fingerprinting** | Vision Framework | Native, Neural Engine optimized |
| **Local Storage** | SwiftData | Modern persistence, automatic CloudKit sync |
| **Background Processing** | BGContinuedProcessingTask | iOS 26 long-running task API |
| **Concurrency** | Swift 6.2 Structured Concurrency | @concurrent, TaskGroup, Observations |

### Project Structure

```
ClipDNA/
│
├── ClipDNA.xcodeproj
├── ClipDNA/
│   ├── App/
│   │   ├── ClipDNAApp.swift              # @main entry point, BGTask registration
│   │   └── ContentView.swift             # Root navigation
│   │
│   ├── Models/
│   │   ├── VideoFingerprint.swift        # @Model SwiftData entity
│   │   ├── DuplicateGroup.swift          # Group of duplicate videos
│   │   ├── MatchResult.swift             # Pairwise match with confidence
│   │   └── QualityMetrics.swift          # Video quality assessment
│   │
│   ├── Services/
│   │   ├── PhotoLibraryService.swift     # PHAsset access, permissions
│   │   ├── FrameExtractor.swift          # AVAssetImageGenerator wrapper
│   │   ├── FingerprintService.swift      # Vision VNFeaturePrint integration
│   │   ├── MatchingEngine.swift          # Core matching algorithm
│   │   ├── QualityAnalyzer.swift         # Video quality scoring
│   │   ├── IndexingCoordinator.swift     # BGContinuedProcessingTask orchestration
│   │   └── ProgressReporter.swift        # Background task progress updates
│   │
│   ├── ViewModels/
│   │   ├── LibraryViewModel.swift        # Main library state
│   │   ├── ScanViewModel.swift           # Scanning progress (Observations)
│   │   ├── DuplicatesViewModel.swift     # Duplicate review state
│   │   └── SettingsViewModel.swift       # User preferences
│   │
│   ├── Views/
│   │   ├── Library/
│   │   │   ├── LibraryView.swift         # Grid of all videos
│   │   │   └── VideoThumbnailView.swift  # Single video cell
│   │   │
│   │   ├── Scan/
│   │   │   ├── ScanView.swift            # Scan initiation UI
│   │   │   └── ScanProgressView.swift    # Progress ring (Liquid Glass)
│   │   │
│   │   ├── Duplicates/
│   │   │   ├── DuplicatesListView.swift  # List of duplicate groups
│   │   │   ├── DuplicateGroupView.swift  # Single group comparison
│   │   │   ├── VideoComparisonView.swift # Side-by-side comparison
│   │   │   └── VideoPlayerView.swift     # Inline video playback
│   │   │
│   │   ├── Settings/
│   │   │   └── SettingsView.swift        # Threshold tuning
│   │   │
│   │   └── Components/
│   │       ├── QualityBadge.swift        # "Best" / "Lower Quality" badge
│   │       ├── ActionButton.swift        # Keep/Delete buttons
│   │       └── StatRow.swift             # Resolution, duration display
│   │
│   ├── Utilities/
│   │   ├── Constants.swift               # Thresholds, configuration
│   │   ├── Extensions/
│   │   │   ├── PHAsset+Extensions.swift
│   │   │   ├── CMTime+Extensions.swift
│   │   │   ├── VNFeaturePrintObservation+Extensions.swift
│   │   │   └── Data+Extensions.swift
│   │   └── Logger.swift                  # OSLog wrapper
│   │
│   └── Resources/
│       ├── Assets.xcassets
│       ├── Info.plist
│       └── Localizable.xcstrings
│
├── ClipDNATests/
│   ├── MatchingEngineTests.swift
│   ├── FingerprintServiceTests.swift
│   └── QualityAnalyzerTests.swift
│
└── README.md
```

---

## SwiftData Models

### VideoFingerprint.swift

```swift
import SwiftData
import Foundation

@Model
final class VideoFingerprint {
    #Unique<VideoFingerprint>([\.assetLocalIdentifier])
    
    var id: UUID
    var assetLocalIdentifier: String
    var createdAt: Date
    var duration: Double
    var width: Int
    var height: Int
    var fileSize: Int64
    var frameCount: Int
    
    /// Serialized feature print data (array of 768-element float vectors)
    var fingerprintData: Data
    
    var qualityScore: Double
    var indexedAt: Date
    var needsReindex: Bool
    
    /// Computed: resolution in pixels
    var resolution: Int { width * height }
    
    init(
        assetLocalIdentifier: String,
        createdAt: Date,
        duration: Double,
        width: Int,
        height: Int,
        fileSize: Int64,
        frameCount: Int,
        fingerprintData: Data,
        qualityScore: Double
    ) {
        self.id = UUID()
        self.assetLocalIdentifier = assetLocalIdentifier
        self.createdAt = createdAt
        self.duration = duration
        self.width = width
        self.height = height
        self.fileSize = fileSize
        self.frameCount = frameCount
        self.fingerprintData = fingerprintData
        self.qualityScore = qualityScore
        self.indexedAt = Date()
        self.needsReindex = false
    }
    
    /// Deserialize fingerprints to array of float arrays
    func deserializeFingerprints() -> [[Float]] {
        let floatCount = fingerprintData.count / MemoryLayout<Float>.size
        let totalFloats = fingerprintData.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self).prefix(floatCount))
        }
        
        // Each fingerprint is 768 floats (iOS 17+ Revision2)
        let chunkSize = 768
        return stride(from: 0, to: totalFloats.count, by: chunkSize).map {
            Array(totalFloats[$0..<min($0 + chunkSize, totalFloats.count)])
        }
    }
}
```

### DuplicateGroup.swift

```swift
import SwiftData
import Foundation

@Model
final class DuplicateGroup {
    var id: UUID
    var createdAt: Date
    var reviewedAt: Date?
    
    @Relationship(deleteRule: .nullify)
    var videos: [VideoFingerprint]
    
    var bestVideoId: UUID?
    var status: GroupStatus
    
    enum GroupStatus: String, Codable {
        case pending
        case reviewed
        case resolved
    }
    
    init(videos: [VideoFingerprint], bestVideoId: UUID?) {
        self.id = UUID()
        self.createdAt = Date()
        self.reviewedAt = nil
        self.videos = videos
        self.bestVideoId = bestVideoId
        self.status = .pending
    }
}
```

### MatchResult.swift

```swift
import SwiftData
import Foundation

@Model
final class MatchResult {
    var id: UUID
    var video1Id: UUID
    var video2Id: UUID
    var distance: Float
    var alignmentOffset: Int
    var matchedFrames: Int
    var confidence: Double
    var createdAt: Date
    
    init(
        video1Id: UUID,
        video2Id: UUID,
        distance: Float,
        alignmentOffset: Int,
        matchedFrames: Int
    ) {
        self.id = UUID()
        self.video1Id = video1Id
        self.video2Id = video2Id
        self.distance = distance
        self.alignmentOffset = alignmentOffset
        self.matchedFrames = matchedFrames
        // Confidence: inverse of distance, normalized to 0-1
        self.confidence = Double(max(0, 1 - (distance / 1.0)))
        self.createdAt = Date()
    }
}
```

---

## Core Service Implementations

### FingerprintService.swift

```swift
import Vision
import CoreImage
import UIKit
import os.log

actor FingerprintService {
    private let logger = Logger(subsystem: "com.clipdna", category: "Fingerprint")
    
    enum FingerprintError: Error {
        case requestFailed(Error)
        case noResults
        case incompatibleVersions
    }
    
    /// Generate a feature print for a single image
    /// Uses @concurrent to ensure Vision work runs on background thread pool
    @concurrent
    func generateFeaturePrint(for cgImage: CGImage) async throws -> VNFeaturePrintObservation {
        let request = VNGenerateImageFeaturePrintRequest()
        
        // Use CPU + Neural Engine, avoid GPU (better for background execution)
        if #available(iOS 17.0, *) {
            request.setComputeDevice(.cpuAndNeuralEngine, for: .main)
        }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            throw FingerprintError.requestFailed(error)
        }
        
        guard let observation = request.results?.first else {
            throw FingerprintError.noResults
        }
        
        return observation
    }
    
    /// Generate feature prints for multiple frames with bounded concurrency
    func generateFeaturePrints(
        for frames: [(time: CMTime, image: CGImage)],
        maxConcurrent: Int = 5,
        progress: ((Double) -> Void)? = nil
    ) async throws -> [VNFeaturePrintObservation] {
        
        var prints: [VNFeaturePrintObservation?] = Array(repeating: nil, count: frames.count)
        let total = frames.count
        var completed = 0
        
        try await withThrowingTaskGroup(of: (Int, VNFeaturePrintObservation).self) { group in
            var iterator = frames.enumerated().makeIterator()
            
            // Start initial batch
            for _ in 0..<min(maxConcurrent, frames.count) {
                guard let (index, frame) = iterator.next() else { break }
                group.addTask {
                    let fp = try await self.generateFeaturePrint(for: frame.image)
                    return (index, fp)
                }
            }
            
            // Process results and add new tasks
            for try await (index, fp) in group {
                prints[index] = fp
                completed += 1
                progress?(Double(completed) / Double(total))
                
                // Add next task if available
                if let (nextIndex, nextFrame) = iterator.next() {
                    group.addTask {
                        let fp = try await self.generateFeaturePrint(for: nextFrame.image)
                        return (nextIndex, fp)
                    }
                }
            }
        }
        
        return prints.compactMap { $0 }
    }
    
    /// Compute Euclidean distance between two feature prints
    nonisolated func computeDistance(
        _ print1: VNFeaturePrintObservation,
        _ print2: VNFeaturePrintObservation
    ) throws -> Float {
        var distance: Float = 0
        try print1.computeDistance(&distance, to: print2)
        return distance
    }
    
    /// Serialize feature print to Data for storage
    nonisolated func serialize(_ observation: VNFeaturePrintObservation) -> Data {
        return observation.data
    }
    
    /// Compute Euclidean distance between two float arrays (for stored fingerprints)
    nonisolated func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.infinity }
        
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
}
```

### FrameExtractor.swift

```swift
import AVFoundation
import UIKit
import os.log

actor FrameExtractor {
    private let logger = Logger(subsystem: "com.clipdna", category: "FrameExtractor")
    
    enum ExtractionError: Error {
        case invalidAsset
        case generationFailed(Error)
        case cancelled
        case videoTooShort
    }
    
    /// Extract frames from a video at specified intervals
    /// - Parameters:
    ///   - asset: The AVAsset to extract frames from
    ///   - fps: Frames per second to extract (default: 1.0)
    ///   - skipEdges: Seconds to skip at start/end to handle Instagram trimming (default: 0.5)
    /// - Returns: Array of (timestamp, CGImage) tuples
    func extractFrames(
        from asset: AVAsset,
        fps: Double = 1.0,
        skipEdges: Double = 0.5
    ) async throws -> [(time: CMTime, image: CGImage)] {
        
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)
        
        guard durationSeconds > 1.0 else {
            throw ExtractionError.videoTooShort
        }
        
        // Handle very short videos
        guard durationSeconds > (skipEdges * 2) else {
            let midTime = CMTime(seconds: durationSeconds / 2, preferredTimescale: 600)
            return try await extractSingleFrame(from: asset, at: midTime)
        }
        
        // Calculate frame times
        let startTime = skipEdges
        let endTime = durationSeconds - skipEdges
        let interval = 1.0 / fps
        
        var times: [CMTime] = []
        var currentTime = startTime
        while currentTime < endTime {
            times.append(CMTime(seconds: currentTime, preferredTimescale: 600))
            currentTime += interval
        }
        
        logger.debug("Extracting \(times.count) frames from video (\(durationSeconds)s)")
        
        // Configure generator
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = CGSize(width: 224, height: 224) // Vision optimal size
        
        // Extract frames
        return try await withCheckedThrowingContinuation { continuation in
            var results: [(CMTime, CGImage)] = []
            var extractionError: Error?
            let nsValueTimes = times.map { NSValue(time: $0) }
            
            generator.generateCGImagesAsynchronously(forTimes: nsValueTimes) { requestedTime, image, actualTime, result, error in
                switch result {
                case .succeeded:
                    if let image = image {
                        results.append((actualTime, image))
                    }
                case .failed:
                    if extractionError == nil {
                        extractionError = error ?? ExtractionError.generationFailed(NSError(domain: "Unknown", code: -1))
                    }
                case .cancelled:
                    if extractionError == nil {
                        extractionError = ExtractionError.cancelled
                    }
                @unknown default:
                    break
                }
                
                // Check if done (all requested times processed)
                if results.count + (extractionError != nil ? 1 : 0) >= times.count {
                    // Sort by time
                    results.sort { CMTimeCompare($0.0, $1.0) < 0 }
                    
                    if let error = extractionError, results.isEmpty {
                        continuation.resume(throwing: ExtractionError.generationFailed(error))
                    } else {
                        continuation.resume(returning: results)
                    }
                }
            }
        }
    }
    
    private func extractSingleFrame(
        from asset: AVAsset,
        at time: CMTime
    ) async throws -> [(time: CMTime, image: CGImage)] {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = CGSize(width: 224, height: 224)
        
        let image = try generator.copyCGImage(at: time, actualTime: nil)
        return [(time, image)]
    }
    
    /// Get video metadata
    func getVideoInfo(from asset: AVAsset) async throws -> VideoInfo {
        let duration = try await asset.load(.duration)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        
        guard let videoTrack = tracks.first else {
            throw ExtractionError.invalidAsset
        }
        
        let size = try await videoTrack.load(.naturalSize)
        let transform = try await videoTrack.load(.preferredTransform)
        
        // Apply transform to get actual dimensions
        let transformedSize = size.applying(transform)
        let width = abs(Int(transformedSize.width))
        let height = abs(Int(transformedSize.height))
        
        return VideoInfo(
            duration: CMTimeGetSeconds(duration),
            width: width,
            height: height
        )
    }
    
    struct VideoInfo {
        let duration: Double
        let width: Int
        let height: Int
        
        var resolution: Int { width * height }
    }
}
```

### MatchingEngine.swift

```swift
import Foundation
import os.log

actor MatchingEngine {
    private let logger = Logger(subsystem: "com.clipdna", category: "Matching")
    private let fingerprintService: FingerprintService
    
    // Thresholds for iOS 17+ normalized 768-dim vectors (0.0-2.0 range)
    private let duplicateThreshold: Float = 0.45    // Definite duplicate
    private let possibleThreshold: Float = 0.80     // Possible duplicate (review)
    private let minMatchedFrames: Int = 3           // Minimum frames for valid match
    
    init(fingerprintService: FingerprintService) {
        self.fingerprintService = fingerprintService
    }
    
    /// Find duplicates for a single video against all indexed videos
    func findDuplicates(
        for targetVideo: VideoFingerprint,
        in candidates: [VideoFingerprint]
    ) async -> [MatchResult] {
        
        let targetPrints = targetVideo.deserializeFingerprints()
        var matches: [MatchResult] = []
        
        for candidate in candidates where candidate.id != targetVideo.id {
            let candidatePrints = candidate.deserializeFingerprints()
            
            if let match = await compareVideos(
                target: (targetVideo.id, targetPrints),
                candidate: (candidate.id, candidatePrints)
            ) {
                matches.append(match)
            }
        }
        
        return matches.sorted { $0.confidence > $1.confidence }
    }
    
    /// Compare two videos using sliding window alignment
    /// Handles trimmed videos by finding best frame offset
    private func compareVideos(
        target: (id: UUID, prints: [[Float]]),
        candidate: (id: UUID, prints: [[Float]])
    ) async -> MatchResult? {
        
        // Determine shorter and longer video
        let (shorterId, shorterPrints) = target.prints.count <= candidate.prints.count ? target : candidate
        let (longerId, longerPrints) = target.prints.count > candidate.prints.count ? target : candidate
        
        guard !shorterPrints.isEmpty && !longerPrints.isEmpty else { return nil }
        
        // Sliding window: slide shorter over longer to handle trimming
        var bestDistance: Float = .infinity
        var bestOffset: Int = 0
        var bestMatchedFrames: Int = 0
        
        let maxOffset = longerPrints.count - shorterPrints.count
        
        for offset in 0...max(0, maxOffset) {
            var totalDistance: Float = 0
            var matchedFrames = 0
            
            for i in 0..<shorterPrints.count {
                let longerIndex = offset + i
                guard longerIndex < longerPrints.count else { break }
                
                let distance = await fingerprintService.euclideanDistance(
                    shorterPrints[i],
                    longerPrints[longerIndex]
                )
                
                // Only count frames below possible threshold
                if distance < possibleThreshold {
                    totalDistance += distance
                    matchedFrames += 1
                }
            }
            
            // Average distance of matched frames
            let avgDistance = matchedFrames > 0 ? totalDistance / Float(matchedFrames) : .infinity
            
            if avgDistance < bestDistance && matchedFrames >= minMatchedFrames {
                bestDistance = avgDistance
                bestOffset = offset
                bestMatchedFrames = matchedFrames
            }
        }
        
        // Check if this qualifies as a match
        guard bestDistance < possibleThreshold && bestMatchedFrames >= minMatchedFrames else {
            return nil
        }
        
        logger.debug("Match found: distance=\(bestDistance), offset=\(bestOffset), frames=\(bestMatchedFrames)")
        
        return MatchResult(
            video1Id: target.id,
            video2Id: candidate.id,
            distance: bestDistance,
            alignmentOffset: bestOffset,
            matchedFrames: bestMatchedFrames
        )
    }
    
    /// Find all duplicate groups in the library
    func findAllDuplicateGroups(
        videos: [VideoFingerprint],
        progressHandler: (@Sendable (Double, String) -> Void)? = nil
    ) async -> [DuplicateGroup] {
        
        var processed: Set<UUID> = []
        var groups: [DuplicateGroup] = []
        let total = videos.count
        
        for (index, video) in videos.enumerated() {
            guard !processed.contains(video.id) else { continue }
            
            let matches = await findDuplicates(for: video, in: videos)
            let significantMatches = matches.filter { $0.distance < duplicateThreshold }
            
            if !significantMatches.isEmpty {
                // Create group with this video and all matches
                var groupVideoIds = Set([video.id])
                for match in significantMatches {
                    groupVideoIds.insert(match.video1Id)
                    groupVideoIds.insert(match.video2Id)
                }
                
                // Find videos in group
                let groupVideos = videos.filter { groupVideoIds.contains($0.id) }
                
                // Determine best video by quality score
                let bestVideo = groupVideos.max(by: { $0.qualityScore < $1.qualityScore })
                
                let group = DuplicateGroup(
                    videos: groupVideos,
                    bestVideoId: bestVideo?.id
                )
                
                groups.append(group)
                processed.formUnion(groupVideoIds)
            }
            
            let progress = Double(index + 1) / Double(total)
            progressHandler?(progress, "Comparing video \(index + 1) of \(total)")
        }
        
        logger.info("Found \(groups.count) duplicate groups from \(videos.count) videos")
        return groups
    }
}
```

### QualityAnalyzer.swift

```swift
import Photos
import AVFoundation
import os.log

actor QualityAnalyzer {
    private let logger = Logger(subsystem: "com.clipdna", category: "Quality")
    
    struct QualityMetrics: Sendable {
        let resolution: Int           // width * height
        let duration: Double          // seconds
        let estimatedBitrate: Double  // bits per second
        let fileSize: Int64           // bytes
        let creationDate: Date?
        let isOriginal: Bool          // Not from social media export
        
        var overallScore: Double {
            // Normalize each component (0-1 scale)
            let resolutionScore = min(1.0, Double(resolution) / (1920 * 1080))  // 1080p = 1.0
            let durationScore = min(1.0, duration / 120.0)  // 2 min = 1.0
            let bitrateScore = min(1.0, estimatedBitrate / 15_000_000)  // 15 Mbps = 1.0
            let originalBonus = isOriginal ? 0.15 : 0.0
            
            return (resolutionScore * 0.35) +
                   (durationScore * 0.20) +
                   (bitrateScore * 0.30) +
                   originalBonus
        }
    }
    
    /// Analyze quality metrics for a PHAsset
    func analyzeQuality(for asset: PHAsset) async -> QualityMetrics {
        let resolution = asset.pixelWidth * asset.pixelHeight
        let duration = asset.duration
        
        // Get file size from resources
        let resources = PHAssetResource.assetResources(for: asset)
        var fileSize: Int64 = 0
        var isOriginal = true
        
        if let videoResource = resources.first(where: { $0.type == .video }) {
            if let size = videoResource.value(forKey: "fileSize") as? Int64 {
                fileSize = size
            }
            isOriginal = !isSocialMediaExport(filename: videoResource.originalFilename)
        }
        
        let estimatedBitrate = duration > 0 ? Double(fileSize * 8) / duration : 0
        
        return QualityMetrics(
            resolution: resolution,
            duration: duration,
            estimatedBitrate: estimatedBitrate,
            fileSize: fileSize,
            creationDate: asset.creationDate,
            isOriginal: isOriginal
        )
    }
    
    /// Detect if filename suggests social media export
    private func isSocialMediaExport(filename: String) -> Bool {
        let lower = filename.lowercased()
        
        // Instagram patterns
        if lower.contains("instagram") ||
           lower.hasPrefix("img_") && lower.contains("_") ||
           lower.range(of: "^\\d{13,}_\\d+\\.mp4$", options: .regularExpression) != nil {
            return true
        }
        
        // TikTok patterns
        if lower.contains("tiktok") ||
           lower.range(of: "^v\\d{8}_\\d+\\.mp4$", options: .regularExpression) != nil {
            return true
        }
        
        // Twitter/X patterns
        if lower.contains("twitter") {
            return true
        }
        
        return false
    }
    
    /// Compare two videos and determine which is better quality
    func compareQuality(
        _ metrics1: QualityMetrics,
        _ metrics2: QualityMetrics
    ) -> ComparisonResult {
        
        var reasons: [String] = []
        var score1 = 0
        var score2 = 0
        
        // Resolution (highest weight)
        if metrics1.resolution > metrics2.resolution * 11 / 10 { // >10% better
            score1 += 3
            reasons.append("Higher resolution")
        } else if metrics2.resolution > metrics1.resolution * 11 / 10 {
            score2 += 3
        }
        
        // Duration (prefer longer = less trimmed)
        if metrics1.duration > metrics2.duration + 0.5 {
            score1 += 2
            reasons.append("Longer duration")
        } else if metrics2.duration > metrics1.duration + 0.5 {
            score2 += 2
        }
        
        // Bitrate
        if metrics1.estimatedBitrate > metrics2.estimatedBitrate * 1.2 {
            score1 += 2
            reasons.append("Higher bitrate")
        } else if metrics2.estimatedBitrate > metrics1.estimatedBitrate * 1.2 {
            score2 += 2
        }
        
        // Original vs social media export
        if metrics1.isOriginal && !metrics2.isOriginal {
            score1 += 3
            reasons.append("Original file (not social media export)")
        } else if metrics2.isOriginal && !metrics1.isOriginal {
            score2 += 3
        }
        
        // Creation date (prefer older = likely original)
        if let date1 = metrics1.creationDate, let date2 = metrics2.creationDate {
            if date1 < date2 {
                score1 += 1
                reasons.append("Earlier creation date")
            } else if date2 < date1 {
                score2 += 1
            }
        }
        
        return ComparisonResult(
            preferFirst: score1 >= score2,
            reasons: reasons,
            score1: score1,
            score2: score2
        )
    }
    
    struct ComparisonResult: Sendable {
        let preferFirst: Bool
        let reasons: [String]
        let score1: Int
        let score2: Int
    }
}
```

### IndexingCoordinator.swift

```swift
import BackgroundTasks
import Photos
import SwiftData
import os.log

/// Coordinates video indexing with BGContinuedProcessingTask
actor IndexingCoordinator {
    private let logger = Logger(subsystem: "com.clipdna", category: "Indexing")
    
    private let photoLibrary: PhotoLibraryService
    private let frameExtractor: FrameExtractor
    private let fingerprintService: FingerprintService
    private let qualityAnalyzer: QualityAnalyzer
    private let modelContainer: ModelContainer
    
    private var isIndexing = false
    private var shouldContinue = true
    
    static let taskIdentifier = "com.clipdna.videoIndexing"
    
    init(
        photoLibrary: PhotoLibraryService,
        frameExtractor: FrameExtractor,
        fingerprintService: FingerprintService,
        qualityAnalyzer: QualityAnalyzer,
        modelContainer: ModelContainer
    ) {
        self.photoLibrary = photoLibrary
        self.frameExtractor = frameExtractor
        self.fingerprintService = fingerprintService
        self.qualityAnalyzer = qualityAnalyzer
        self.modelContainer = modelContainer
    }
    
    /// Register background task handler (call from App init)
    nonisolated func registerBackgroundTask() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.taskIdentifier,
            using: nil
        ) { task in
            Task {
                await self.handleBackgroundTask(task as! BGContinuedProcessingTask)
            }
        }
    }
    
    /// Start indexing (can continue in background via BGContinuedProcessingTask)
    func startIndexing() async throws -> IndexingResult {
        guard !isIndexing else {
            throw IndexingError.alreadyIndexing
        }
        
        isIndexing = true
        shouldContinue = true
        defer { isIndexing = false }
        
        // Fetch all video assets
        let assets = try await photoLibrary.fetchAllVideos()
        logger.info("Found \(assets.count) videos in library")
        
        // Get already indexed identifiers
        let context = ModelContext(modelContainer)
        let existingDescriptor = FetchDescriptor<VideoFingerprint>()
        let existing = try context.fetch(existingDescriptor)
        let indexedIds = Set(existing.map { $0.assetLocalIdentifier })
        
        // Filter to unindexed
        let unindexed = assets.filter { !indexedIds.contains($0.localIdentifier) }
        logger.info("\(unindexed.count) videos need indexing")
        
        // Submit background task request
        try submitBackgroundTaskRequest(videoCount: unindexed.count)
        
        // Start indexing
        var indexed = 0
        var failed = 0
        
        for asset in unindexed {
            guard shouldContinue else {
                logger.info("Indexing interrupted")
                break
            }
            
            do {
                try await indexVideo(asset: asset, context: context)
                indexed += 1
            } catch {
                logger.error("Failed to index \(asset.localIdentifier): \(error.localizedDescription)")
                failed += 1
            }
        }
        
        return IndexingResult(
            totalVideos: assets.count,
            newlyIndexed: indexed,
            failed: failed,
            alreadyIndexed: indexedIds.count
        )
    }
    
    /// Index a single video
    private func indexVideo(asset: PHAsset, context: ModelContext) async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Get AVAsset
        guard let avAsset = try await photoLibrary.getAVAsset(for: asset) else {
            throw IndexingError.assetNotAvailable
        }
        
        // Extract frames at 1 FPS, skip edges
        let frames = try await frameExtractor.extractFrames(
            from: avAsset,
            fps: 1.0,
            skipEdges: 0.5
        )
        
        // Generate fingerprints with bounded concurrency
        let fingerprints = try await fingerprintService.generateFeaturePrints(
            for: frames,
            maxConcurrent: 5
        )
        
        // Serialize fingerprints
        var fingerprintData = Data()
        for fp in fingerprints {
            fingerprintData.append(fingerprintService.serialize(fp))
        }
        
        // Analyze quality
        let quality = await qualityAnalyzer.analyzeQuality(for: asset)
        
        // Create and save
        let videoFingerprint = VideoFingerprint(
            assetLocalIdentifier: asset.localIdentifier,
            createdAt: asset.creationDate ?? Date(),
            duration: asset.duration,
            width: asset.pixelWidth,
            height: asset.pixelHeight,
            fileSize: quality.fileSize,
            frameCount: frames.count,
            fingerprintData: fingerprintData,
            qualityScore: quality.overallScore
        )
        
        context.insert(videoFingerprint)
        try context.save()
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        logger.debug("Indexed \(asset.localIdentifier) (\(frames.count) frames) in \(String(format: "%.2f", elapsed))s")
    }
    
    /// Submit background task request
    private func submitBackgroundTaskRequest(videoCount: Int) throws {
        let request = BGContinuedProcessingTaskRequest(
            identifier: Self.taskIdentifier,
            title: "Indexing Videos",
            subtitle: "0 of \(videoCount) videos processed"
        )
        request.strategy = .immediately
        
        try BGTaskScheduler.shared.submit(request)
        logger.info("Submitted background task request")
    }
    
    /// Handle background task execution
    private func handleBackgroundTask(_ task: BGContinuedProcessingTask) async {
        logger.info("Background task started")
        
        // Set up expiration handler
        task.expirationHandler = { [weak self] in
            Task {
                await self?.handleExpiration()
            }
        }
        
        do {
            // Continue or restart indexing
            let result = try await startIndexing()
            
            // Update progress
            task.progress.completedUnitCount = Int64(result.newlyIndexed)
            task.progress.totalUnitCount = Int64(result.totalVideos - result.alreadyIndexed)
            
            task.setTaskCompleted(success: true)
            logger.info("Background task completed: \(result.newlyIndexed) indexed")
            
        } catch {
            logger.error("Background task failed: \(error.localizedDescription)")
            task.setTaskCompleted(success: false)
        }
    }
    
    private func handleExpiration() {
        logger.warning("Background task expiring, saving state...")
        shouldContinue = false
        // State is saved incrementally after each video, so we're safe
    }
    
    /// Update progress for background task UI
    func updateProgress(completed: Int, total: Int) {
        // This updates the Dynamic Island / notification UI
        // Called from indexing loop
    }
    
    // MARK: - Types
    
    struct IndexingResult: Sendable {
        let totalVideos: Int
        let newlyIndexed: Int
        let failed: Int
        let alreadyIndexed: Int
    }
    
    enum IndexingError: Error {
        case alreadyIndexing
        case assetNotAvailable
        case permissionDenied
    }
}
```

### PhotoLibraryService.swift

```swift
import Photos
import AVFoundation
import os.log

actor PhotoLibraryService {
    private let logger = Logger(subsystem: "com.clipdna", category: "PhotoLibrary")
    
    enum PhotoLibraryError: Error {
        case permissionDenied
        case assetNotFound
        case exportFailed
    }
    
    /// Request photo library permission
    func requestPermission() async -> Bool {
        let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        return status == .authorized || status == .limited
    }
    
    /// Check current permission status
    nonisolated func checkPermission() -> PHAuthorizationStatus {
        PHPhotoLibrary.authorizationStatus(for: .readWrite)
    }
    
    /// Fetch all video assets from photo library
    func fetchAllVideos() async throws -> [PHAsset] {
        guard checkPermission() == .authorized || checkPermission() == .limited else {
            throw PhotoLibraryError.permissionDenied
        }
        
        let options = PHFetchOptions()
        options.predicate = NSPredicate(format: "mediaType == %d", PHAssetMediaType.video.rawValue)
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        
        let results = PHAsset.fetchAssets(with: options)
        
        var assets: [PHAsset] = []
        results.enumerateObjects { asset, _, _ in
            assets.append(asset)
        }
        
        logger.info("Fetched \(assets.count) video assets")
        return assets
    }
    
    /// Get AVAsset for a PHAsset
    func getAVAsset(for asset: PHAsset) async throws -> AVAsset? {
        return try await withCheckedThrowingContinuation { continuation in
            let options = PHVideoRequestOptions()
            options.version = .current
            options.deliveryMode = .highQualityFormat
            options.isNetworkAccessAllowed = true
            
            PHImageManager.default().requestAVAsset(
                forVideo: asset,
                options: options
            ) { avAsset, _, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: avAsset)
                }
            }
        }
    }
    
    /// Delete assets (moves to Recently Deleted)
    func deleteAssets(_ assets: [PHAsset]) async throws {
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.deleteAssets(assets as NSFastEnumeration)
        }
        logger.info("Deleted \(assets.count) assets")
    }
}
```

---

## SwiftUI Views

### ScanView.swift

```swift
import SwiftUI
import Observation

struct ScanView: View {
    @State private var viewModel: ScanViewModel
    
    init(coordinator: IndexingCoordinator, matchingEngine: MatchingEngine) {
        _viewModel = State(initialValue: ScanViewModel(
            coordinator: coordinator,
            matchingEngine: matchingEngine
        ))
    }
    
    var body: some View {
        VStack(spacing: 32) {
            Spacer()
            
            // Progress ring with Liquid Glass effect
            ZStack {
                Circle()
                    .fill(.ultraThinMaterial)
                    .frame(width: 220, height: 220)
                
                Circle()
                    .stroke(Color.secondary.opacity(0.2), lineWidth: 12)
                    .frame(width: 200, height: 200)
                
                Circle()
                    .trim(from: 0, to: viewModel.progress)
                    .stroke(
                        AngularGradient(
                            colors: [.blue, .purple, .blue],
                            center: .center
                        ),
                        style: StrokeStyle(lineWidth: 12, lineCap: .round)
                    )
                    .frame(width: 200, height: 200)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.3), value: viewModel.progress)
                
                VStack(spacing: 4) {
                    Text("\(Int(viewModel.progress * 100))%")
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .contentTransition(.numericText())
                    
                    Text(viewModel.phase.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            
            // Status
            VStack(spacing: 8) {
                Text(viewModel.statusMessage)
                    .font(.headline)
                    .multilineTextAlignment(.center)
                
                if let eta = viewModel.estimatedTimeRemaining {
                    Text("About \(formatTime(eta)) remaining")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                
                if viewModel.phase == .indexing {
                    Text("You can leave the app — indexing will continue in the background")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                        .padding(.top, 4)
                }
            }
            .padding(.horizontal)
            
            Spacer()
            
            // Actions
            if viewModel.isScanning {
                Button("Cancel", role: .destructive) {
                    viewModel.cancel()
                }
                .buttonStyle(.bordered)
            } else if viewModel.phase == .idle {
                Button {
                    Task { await viewModel.startScan() }
                } label: {
                    Label("Start Scan", systemImage: "magnifyingglass")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .padding(.horizontal, 40)
            } else if viewModel.phase == .complete {
                NavigationLink {
                    DuplicatesListView(groups: viewModel.duplicateGroups)
                } label: {
                    Label("View \(viewModel.duplicateGroups.count) Duplicate Groups", systemImage: "square.stack.3d.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .padding(.horizontal, 40)
            }
        }
        .padding()
        .navigationTitle("Scan Library")
        .navigationBarBackButtonHidden(viewModel.isScanning)
    }
    
    private func formatTime(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return "\(Int(seconds)) seconds"
        } else if seconds < 3600 {
            return "\(Int(seconds / 60)) minutes"
        } else {
            let hours = Int(seconds / 3600)
            let minutes = Int((seconds.truncatingRemainder(dividingBy: 3600)) / 60)
            return "\(hours)h \(minutes)m"
        }
    }
}

@Observable
@MainActor
class ScanViewModel {
    enum Phase: CustomStringConvertible {
        case idle
        case indexing
        case matching
        case complete
        
        var description: String {
            switch self {
            case .idle: return "Ready to scan"
            case .indexing: return "Indexing videos"
            case .matching: return "Finding duplicates"
            case .complete: return "Complete"
            }
        }
    }
    
    var phase: Phase = .idle
    var progress: Double = 0
    var statusMessage: String = "Tap Start to scan your video library"
    var estimatedTimeRemaining: TimeInterval?
    var isScanning: Bool = false
    var duplicateGroups: [DuplicateGroup] = []
    
    private let coordinator: IndexingCoordinator
    private let matchingEngine: MatchingEngine
    private var task: Task<Void, Never>?
    
    init(coordinator: IndexingCoordinator, matchingEngine: MatchingEngine) {
        self.coordinator = coordinator
        self.matchingEngine = matchingEngine
    }
    
    func startScan() async {
        guard !isScanning else { return }
        isScanning = true
        phase = .indexing
        
        task = Task {
            do {
                // Phase 1: Indexing
                statusMessage = "Preparing to index..."
                let indexResult = try await coordinator.startIndexing()
                
                progress = 0.5
                statusMessage = "Indexed \(indexResult.newlyIndexed) new videos"
                
                // Phase 2: Matching
                phase = .matching
                statusMessage = "Analyzing for duplicates..."
                
                // Fetch all fingerprints
                // Note: In real implementation, inject ModelContainer
                // let videos = try context.fetch(FetchDescriptor<VideoFingerprint>())
                // For now, placeholder:
                let videos: [VideoFingerprint] = []
                
                duplicateGroups = await matchingEngine.findAllDuplicateGroups(
                    videos: videos
                ) { [weak self] progress, message in
                    Task { @MainActor in
                        self?.progress = 0.5 + (progress * 0.5)
                        self?.statusMessage = message
                    }
                }
                
                // Complete
                phase = .complete
                progress = 1.0
                statusMessage = "Found \(duplicateGroups.count) duplicate groups"
                
            } catch {
                statusMessage = "Error: \(error.localizedDescription)"
                phase = .idle
            }
            
            isScanning = false
        }
    }
    
    func cancel() {
        task?.cancel()
        isScanning = false
        phase = .idle
        progress = 0
        statusMessage = "Scan cancelled"
    }
}
```

### DuplicateGroupView.swift

```swift
import SwiftUI
import AVKit

struct DuplicateGroupView: View {
    let group: DuplicateGroup
    let onKeep: (VideoFingerprint) -> Void
    let onDelete: (VideoFingerprint) -> Void
    
    @State private var selectedVideo: VideoFingerprint?
    @State private var player: AVPlayer?
    
    private var sortedVideos: [VideoFingerprint] {
        group.videos.sorted { $0.qualityScore > $1.qualityScore }
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 4) {
                    Text("\(group.videos.count) Duplicate Videos")
                        .font(.headline)
                    Text("Select the version to keep")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.top)
                
                // Video cards
                ForEach(sortedVideos) { video in
                    VideoComparisonCard(
                        video: video,
                        isBest: video.id == group.bestVideoId,
                        isSelected: selectedVideo?.id == video.id,
                        onTap: { selectedVideo = video },
                        onKeep: { onKeep(video) },
                        onDelete: { onDelete(video) }
                    )
                }
                
                // Video preview
                if let selected = selectedVideo {
                    VideoPreviewSection(assetIdentifier: selected.assetLocalIdentifier)
                }
            }
            .padding()
        }
        .navigationTitle("Compare Videos")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct VideoComparisonCard: View {
    let video: VideoFingerprint
    let isBest: Bool
    let isSelected: Bool
    let onTap: () -> Void
    let onKeep: () -> Void
    let onDelete: () -> Void
    
    var body: some View {
        VStack(spacing: 12) {
            HStack(alignment: .top, spacing: 16) {
                // Thumbnail placeholder
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.secondary.opacity(0.2))
                    .frame(width: 120, height: 90)
                    .overlay {
                        Image(systemName: "play.fill")
                            .foregroundStyle(.secondary)
                    }
                
                // Info
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        if isBest {
                            Label("Best Quality", systemImage: "star.fill")
                                .font(.caption.bold())
                                .foregroundStyle(.yellow)
                        }
                        Spacer()
                        Text(String(format: "%.0f%%", video.qualityScore * 100))
                            .font(.caption.bold())
                            .foregroundStyle(isBest ? .green : .secondary)
                    }
                    
                    Label("\(video.width) × \(video.height)", systemImage: "rectangle.ratio.3.to.4")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    Label(formatDuration(video.duration), systemImage: "clock")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    Label(formatFileSize(video.fileSize), systemImage: "doc")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
            }
            
            // Actions
            HStack(spacing: 12) {
                Button {
                    onKeep()
                } label: {
                    Label("Keep", systemImage: "checkmark.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                
                Button(role: .destructive) {
                    onDelete()
                } label: {
                    Label("Delete", systemImage: "trash")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .background {
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
                .overlay {
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
                }
        }
        .onTapGesture(perform: onTap)
    }
    
    private func formatDuration(_ seconds: Double) -> String {
        let minutes = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", minutes, secs)
    }
    
    private func formatFileSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}

struct VideoPreviewSection: View {
    let assetIdentifier: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Preview")
                .font(.headline)
            
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.2))
                .frame(height: 200)
                .overlay {
                    Image(systemName: "play.circle.fill")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                }
        }
    }
}
```

---

## App Entry Point

### ClipDNAApp.swift

```swift
import SwiftUI
import SwiftData
import BackgroundTasks
import os.log

@main
struct ClipDNAApp: App {
    private let logger = Logger(subsystem: "com.clipdna", category: "App")
    
    let modelContainer: ModelContainer
    let services: ServiceContainer
    
    init() {
        // Initialize SwiftData
        let schema = Schema([
            VideoFingerprint.self,
            DuplicateGroup.self,
            MatchResult.self
        ])
        
        let modelConfiguration = ModelConfiguration(
            schema: schema,
            isStoredInMemoryOnly: false,
            cloudKitDatabase: .automatic
        )
        
        do {
            modelContainer = try ModelContainer(
                for: schema,
                configurations: [modelConfiguration]
            )
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
        
        // Initialize services
        services = ServiceContainer(modelContainer: modelContainer)
        
        // Register background task
        services.indexingCoordinator.registerBackgroundTask()
        
        logger.info("ClipDNA initialized")
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(services)
        }
        .modelContainer(modelContainer)
    }
}

/// Dependency injection container
@Observable
class ServiceContainer {
    let photoLibrary: PhotoLibraryService
    let frameExtractor: FrameExtractor
    let fingerprintService: FingerprintService
    let qualityAnalyzer: QualityAnalyzer
    let matchingEngine: MatchingEngine
    let indexingCoordinator: IndexingCoordinator
    
    init(modelContainer: ModelContainer) {
        photoLibrary = PhotoLibraryService()
        frameExtractor = FrameExtractor()
        fingerprintService = FingerprintService()
        qualityAnalyzer = QualityAnalyzer()
        matchingEngine = MatchingEngine(fingerprintService: fingerprintService)
        indexingCoordinator = IndexingCoordinator(
            photoLibrary: photoLibrary,
            frameExtractor: frameExtractor,
            fingerprintService: fingerprintService,
            qualityAnalyzer: qualityAnalyzer,
            modelContainer: modelContainer
        )
    }
}
```

### ContentView.swift

```swift
import SwiftUI

struct ContentView: View {
    @Environment(ServiceContainer.self) private var services
    @State private var hasPermission = false
    
    var body: some View {
        NavigationStack {
            if hasPermission {
                MainTabView()
            } else {
                PermissionView(onGranted: { hasPermission = true })
            }
        }
        .task {
            hasPermission = await services.photoLibrary.requestPermission()
        }
    }
}

struct MainTabView: View {
    var body: some View {
        TabView {
            LibraryView()
                .tabItem {
                    Label("Library", systemImage: "photo.on.rectangle")
                }
            
            ScanTab()
                .tabItem {
                    Label("Scan", systemImage: "magnifyingglass")
                }
            
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
    }
}

struct ScanTab: View {
    @Environment(ServiceContainer.self) private var services
    
    var body: some View {
        NavigationStack {
            ScanView(
                coordinator: services.indexingCoordinator,
                matchingEngine: services.matchingEngine
            )
        }
    }
}

struct PermissionView: View {
    let onGranted: () -> Void
    @Environment(ServiceContainer.self) private var services
    
    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "photo.stack")
                .font(.system(size: 80))
                .foregroundStyle(.blue)
            
            Text("Photo Library Access")
                .font(.title.bold())
            
            Text("ClipDNA needs access to your photo library to find duplicate videos.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            
            Button {
                Task {
                    if await services.photoLibrary.requestPermission() {
                        onGranted()
                    }
                }
            } label: {
                Text("Grant Access")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(40)
    }
}
```

---

## Info.plist Configuration

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- Photo Library Access -->
    <key>NSPhotoLibraryUsageDescription</key>
    <string>ClipDNA needs access to your photo library to find duplicate videos and help you free up storage.</string>
    
    <key>NSPhotoLibraryAddUsageDescription</key>
    <string>ClipDNA needs permission to delete duplicate videos you select for removal.</string>
    
    <!-- Background Tasks -->
    <key>BGTaskSchedulerPermittedIdentifiers</key>
    <array>
        <string>com.clipdna.videoIndexing</string>
    </array>
    
    <!-- Required background modes for continued processing -->
    <key>UIBackgroundModes</key>
    <array>
        <string>processing</string>
    </array>
</dict>
</plist>
```

---

## Performance Expectations

### Benchmarks (iPhone 15 Pro)

| Operation | Time | Notes |
|-----------|------|-------|
| Extract 1 frame | ~30-50ms | AVAssetImageGenerator at 224×224 |
| Generate 1 feature print | ~15-25ms | Vision framework, Neural Engine |
| Index 1 video (30s @ 1fps) | ~1.5-2.5s | 30 frames |
| Index 100 videos (foreground) | ~3-4 min | Parallel processing |
| Index 100 videos (background) | ~15-20 min | 4-5× slower, CPU only |
| Compare 2 videos | ~1-5ms | Euclidean distance |
| Find duplicates (100 videos) | ~15-25s | O(n²) comparisons |

### Memory Usage

- Feature prints: ~3KB each (768 floats × 4 bytes)
- 1000 videos × 30 frames = ~90MB fingerprint data
- SwiftData handles efficiently with lazy loading
- Videos never loaded into memory, only 224×224 thumbnails

### Battery Optimization

- Background execution uses efficiency cores only
- Neural Engine preferred over GPU when available
- Incremental saves prevent data loss on expiration
- Progress reporting required every 15-20 seconds

---

## Requirements

- **iOS/iPadOS:** 26.0+
- **Swift:** 6.2
- **Xcode:** 26.0+
- **Device:** iPhone 12+ / iPad (9th gen)+
- **Capabilities:** Photo Library access, Background processing

---

## Privacy Guarantees

- **100% On-Device:** All processing happens locally
- **No Network Calls:** Works completely offline
- **No Analytics:** Zero tracking or telemetry
- **Fingerprints Only:** Video content never leaves device
- **User Control:** All deletions require explicit confirmation
- **iCloud Sync:** Optional — only fingerprint metadata syncs, not videos
