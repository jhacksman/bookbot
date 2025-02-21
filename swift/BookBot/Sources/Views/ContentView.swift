import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var appState: AppState
    
    var body: some View {
        NavigationSplitView {
            LibraryView()
        } detail: {
            if let book = appState.selectedBook {
                BookDetailView(book: book)
            } else {
                Text("Select a book")
                    .foregroundStyle(.secondary)
            }
        }
        .searchable(text: $appState.searchQuery)
        .alert("Error", isPresented: .constant(appState.errorMessage != nil)) {
            Button("OK") {
                appState.errorMessage = nil
            }
        } message: {
            if let error = appState.errorMessage {
                Text(error)
            }
        }
    }
}
