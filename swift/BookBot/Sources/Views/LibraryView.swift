import SwiftUI

struct LibraryView: View {
    @EnvironmentObject private var appState: AppState
    
    var body: some View {
        List(selection: $appState.selectedBook) {
            if appState.isLoading {
                ProgressView()
                    .progressViewStyle(.circular)
            } else {
                ForEach(books) { book in
                    BookRow(book: book)
                }
            }
        }
        .navigationTitle("Library")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: refreshLibrary) {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
        }
    }
    
    private func refreshLibrary() {
        // Implement refresh logic
    }
}
