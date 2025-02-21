import SwiftUI

struct BookDetailView: View {
    let book: Book
    @State private var question = ""
    @State private var isQuerying = false
    @State private var queryResponse: QueryResponse?
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(book.title)
                    .font(.title)
                Text(book.author)
                    .font(.headline)
                    .foregroundStyle(.secondary)
                
                if let description = book.description {
                    Text(description)
                        .font(.body)
                }
                
                Divider()
                
                VStack(alignment: .leading) {
                    TextField("Ask a question about this book...", text: $question)
                        .textFieldStyle(.roundedBorder)
                    
                    Button(action: submitQuery) {
                        if isQuerying {
                            ProgressView()
                        } else {
                            Text("Ask")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(question.isEmpty || isQuerying)
                }
                
                if let response = queryResponse {
                    QueryResponseView(response: response)
                }
            }
            .padding()
        }
    }
    
    private func submitQuery() {
        // Implement query submission
    }
}
