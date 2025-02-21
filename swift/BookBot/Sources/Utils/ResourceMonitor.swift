import Foundation

final class ResourceMonitor {
    static let shared = ResourceMonitor()
    
    private(set) var totalMemory: UInt64 = 0
    private(set) var memoryUsage: UInt64 = 0
    private(set) var cpuUsage: Double = 0
    
    private var timer: Timer?
    
    private init() {
        setupMonitoring()
    }
    
    private func setupMonitoring() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateMetrics()
        }
    }
    
    private func updateMetrics() {
        var stats = host_statistics64_data_t()
        var size = mach_msg_type_number_t(MemoryLayout<host_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let host = mach_host_self()
        
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            host_statistics64(host,
                            HOST_VM_INFO64,
                            pointer.withMemoryRebound(to: host_info64_t.self, capacity: 1) { $0 },
                            &size)
        }
        
        if result == KERN_SUCCESS {
            let used = UInt64(stats.active_count + stats.wire_count) * UInt64(vm_page_size)
            memoryUsage = used
            
            let total = ProcessInfo.processInfo.physicalMemory
            totalMemory = total
            
            cpuUsage = Double(stats.cpu_utilization) / 100.0
        }
    }
    
    deinit {
        timer?.invalidate()
    }
}
