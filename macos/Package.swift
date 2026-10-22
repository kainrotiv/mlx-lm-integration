// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LocalAIApp",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "LocalAIApp", targets: ["LocalAIApp"]),
    ],
    targets: [
        .executableTarget(name: "LocalAIApp"),
    ]
)
