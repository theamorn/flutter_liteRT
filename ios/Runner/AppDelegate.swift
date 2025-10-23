import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
    
    // Setup Method Channel for native camera
    let controller = window?.rootViewController as! FlutterViewController
    let nativeCameraChannel = FlutterMethodChannel(
      name: "com.example.flutter_litert/native_camera",
      binaryMessenger: controller.binaryMessenger
    )
    
    nativeCameraChannel.setMethodCallHandler { [weak controller] (call, result) in
      if call.method == "openNativeCamera" {
        DispatchQueue.main.async {
          guard let flutterController = controller else {
            result(FlutterError(code: "ERROR", message: "Controller not found", details: nil))
            return
          }
          
          // Get Flutter asset keys
          let modelKey = flutterController.lookupKey(forAsset: "assets/food-classifier.tflite")
          let labelsKey = flutterController.lookupKey(forAsset: "assets/aiy_food_V1_labelmap.csv")
          
          let nativeCameraVC = NativeCameraViewController(modelKey: modelKey, labelsKey: labelsKey)
          nativeCameraVC.modalPresentationStyle = .fullScreen
          flutterController.present(nativeCameraVC, animated: true)
          result(nil)
        }
      } else {
        result(FlutterMethodNotImplemented)
      }
    }
    
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
