// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "BookBot",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "BookBot", targets: ["BookBot"])
    ],
    dependencies: [
        .package(url: "https://github.com/stephencelis/SQLite.swift.git", from: "0.14.1"),
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.8.1"),
        .package(url: "https://github.com/SwiftyJSON/SwiftyJSON.git", from: "5.0.1")
    ],
    targets: [
        .executableTarget(
            name: "BookBot",
            dependencies: [
                .product(name: "SQLite", package: "SQLite.swift"),
                .product(name: "Alamofire", package: "Alamofire"),
                .product(name: "SwiftyJSON", package: "SwiftyJSON")
            ],
            path: "Sources"
        ),
        .testTarget(
            name: "BookBotTests",
            dependencies: ["BookBot"],
            path: "Tests"
        )
    ]
)
