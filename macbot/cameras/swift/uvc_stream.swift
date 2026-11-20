// Stream a UVC camera opened by its AVFoundation uniqueID as cropped JPEG frames on stdout.
// Opening by uniqueID avoids grabbing a Continuity Camera (iPhone) by accident.
//
// Frame format: 8-byte little-endian Double (capture time, Unix seconds)
//             + 4-byte little-endian UInt32 (JPEG length) + JPEG bytes.
// Usage: uvc_stream <uniqueID> <x0> <y0> <x1> <y1> [jpeg quality 0-1, default 0.9] [fps, default 30]
//        x1 = 0 streams the full frame. Crop coordinates are top-left based pixels (as in OpenCV).
// The process exits when the parent closes the pipe (SIGPIPE) or on SIGTERM.
import AVFoundation
import CoreImage
import Foundation

func fail(_ msg: String, _ code: Int32) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(code)
}

let a = CommandLine.arguments
func dimensions(_ format: AVCaptureDevice.Format) -> CMVideoDimensions {
    CMVideoFormatDescriptionGetDimensions(format.formatDescription)
}

if a.count == 2 && a[1] == "--list" {
    var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera]
    if #available(macOS 14.0, *) {
        types += [.external, .continuityCamera]
    } else {
        types.append(.externalUnknown)
    }
    let devices = AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .video, position: .unspecified).devices
    let rows: [[String: Any]] = devices.map { dev in
        var continuity = false
        if #available(macOS 14.0, *) { continuity = dev.isContinuityCamera || dev.deviceType == .continuityCamera }
        let modes = dev.formats.flatMap { format in
            let d = dimensions(format)
            return format.videoSupportedFrameRateRanges.compactMap { r -> [Int]? in
                let fps = Int(floor(min(30.001, r.maxFrameRate) + 0.05))
                return fps > 0 && Double(fps) >= r.minFrameRate - 0.05 ? [Int(d.width), Int(d.height), fps] : nil
            }
        }
        return ["name": dev.localizedName, "unique_id": dev.uniqueID, "model": dev.modelID,
                "continuity": continuity, "builtin": dev.deviceType == .builtInWideAngleCamera, "modes": modes]
    }
    FileHandle.standardOutput.write(try! JSONSerialization.data(withJSONObject: rows))
    exit(0)
}

guard a.count >= 6, let x0 = Int(a[2]), let y0 = Int(a[3]), let x1 = Int(a[4]), let y1 = Int(a[5]) else {
    fail("usage: uvc_stream <uniqueID> <x0> <y0> <x1> <y1> [quality] [fps]", 2)
}
let quality = a.count > 6 ? Double(a[6])! : 0.9
let fps = a.count > 7 ? Int32(a[7])! : 30
guard let dev = AVCaptureDevice(uniqueID: a[1]) else { fail("camera not found: \(a[1])", 1) }
let width = a.count > 9 ? Int(a[8]) : nil
let height = a.count > 9 ? Int(a[9]) : nil

final class Out: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let x0: Int, y0: Int, x1: Int, y1: Int, quality: Double
    let ctx = CIContext()
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    var checked = false
    init(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int, _ q: Double) {
        self.x0 = x0; self.y0 = y0; self.x1 = x1; self.y1 = y1; self.quality = q
    }
    func captureOutput(_ o: AVCaptureOutput, didOutput sb: CMSampleBuffer, from c: AVCaptureConnection) {
        let t = Date().timeIntervalSince1970
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let full = x1 == 0
        if !checked {
            checked = true
            if !full && (x1 > w || y1 > h) { fail("frame is \(w)x\(h), crop box exceeds it", 3) }
        }
        var img = CIImage(cvPixelBuffer: pb)
        if !full {  // CoreImage has its origin at the bottom left
            let r = CGRect(x: x0, y: h - y1, width: x1 - x0, height: y1 - y0)
            img = img.cropped(to: r).transformed(by: CGAffineTransform(translationX: -r.minX, y: -r.minY))
        }
        let opts = [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): quality]
        guard let jpg = ctx.jpegRepresentation(of: img, colorSpace: cs, options: opts) else { return }
        var head = Data(count: 12)
        head.withUnsafeMutableBytes { p in
            p.storeBytes(of: t.bitPattern.littleEndian, toByteOffset: 0, as: UInt64.self)
            p.storeBytes(of: UInt32(jpg.count).littleEndian, toByteOffset: 8, as: UInt32.self)
        }
        FileHandle.standardOutput.write(head + jpg)
    }
}

let s = AVCaptureSession()
// Without an explicit mode, keep the legacy 1080p preset when available.
do {
    let input = try AVCaptureDeviceInput(device: dev)
    guard s.canAddInput(input) else { fail("cannot open the selected camera", 1) }
    s.addInput(input)
} catch { fail("cannot open the selected camera: \(error)", 1) }
if let w = width, let h = height {
    guard let format = dev.formats.first(where: { f in
        let d = dimensions(f)
        return Int(d.width) == w && Int(d.height) == h &&
            f.videoSupportedFrameRateRanges.contains { $0.minFrameRate - 0.05 <= Double(fps) && $0.maxFrameRate + 0.05 >= Double(fps) }
    }) else { fail("camera does not support \(w)x\(h) at \(fps) fps", 1) }
    do {
        try dev.lockForConfiguration()
        dev.activeFormat = format
        dev.unlockForConfiguration()
    } catch { fail("cannot select camera format: \(error)", 1) }
} else if s.canSetSessionPreset(.hd1920x1080) { s.sessionPreset = .hd1920x1080 }
let o = AVCaptureVideoDataOutput()
o.alwaysDiscardsLateVideoFrames = true
o.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
let out = Out(x0, y0, x1, y1, quality)
o.setSampleBufferDelegate(out, queue: DispatchQueue(label: "cam"))
s.addOutput(o)
s.startRunning()
do {
    // Many UVC cameras default to 25 fps. Lock the rate using the exact duration the format reports
    // (the 30 fps entry is 1000000/30000030 s, not 1/30; anything else throws).
    try dev.lockForConfiguration()
    if let r = dev.activeFormat.videoSupportedFrameRateRanges.min(by: {
        abs($0.maxFrameRate - Double(fps)) < abs($1.maxFrameRate - Double(fps)) }) {
        let rate = Double(fps)
        // The legacy path uses the fastest duration in the selected range.
        let duration = width == nil || rate >= r.maxFrameRate ? r.minFrameDuration :
            rate <= r.minFrameRate ? r.maxFrameDuration : CMTime(value: 1, timescale: fps)
        dev.activeVideoMinFrameDuration = duration
        dev.activeVideoMaxFrameDuration = duration
    }
    dev.unlockForConfiguration()
} catch {
    FileHandle.standardError.write("could not set frame rate: \(error)\n".data(using: .utf8)!)
}
signal(SIGTERM) { _ in exit(0) }
RunLoop.main.run()
