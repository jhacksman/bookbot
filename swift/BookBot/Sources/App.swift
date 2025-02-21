import SwiftUI
import Combine

@main
struct BookBotApp: App {
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}

final class AppState: ObservableObject {
    @Published var isLoading = false
    @Published var selectedBook: Book?
    @Published var searchQuery = ""
    @Published var errorMessage: String?
    
    private let apiClient = APIClient()
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        setupBindings()
        monitorResources()
    }
    
    private func setupBindings() {
        $searchQuery
            .debounce(for: .milliseconds(300), scheduler: DispatchQueue.main)
            .sink { [weak self] query in
                guard !query.isEmpty else { return }
                self?.performSearch(query: query)
            }
            .store(in: &cancellables)
    }
    
    private func monitorResources() {
        Timer.publish(every: 5.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.checkSystemResources()
            }
            .store(in: &cancellables)
    }
    
    private func performSearch(query: String) {
        isLoading = true
        apiClient.search(query: query)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [weak self] completion in
                    self?.isLoading = false
                    if case .failure(let error) = completion {
                        self?.errorMessage = error.localizedDescription
                    }
                },
                receiveValue: { [weak self] books in
                    self?.updateSearchResults(books)
                }
            )
            .store(in: &cancellables)
    }
    
    private func checkSystemResources() {
        let memoryUsage = ResourceMonitor.shared.memoryUsage
        if memoryUsage > 0.8 * ResourceMonitor.shared.totalMemory {
            errorMessage = "High memory usage detected"
        }
    }
}
