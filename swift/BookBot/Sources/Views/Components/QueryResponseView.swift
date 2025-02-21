import SwiftUI

struct QueryResponseView: View {
    let response: QueryResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(response.answer)
                .font(.body)
            
            if !response.citations.isEmpty {
                Text("Citations")
                    .font(.headline)
                
                ForEach(response.citations) { citation in
                    CitationView(citation: citation)
                }
            }
            
            HStack {
                Text("Confidence:")
                ProgressView(value: response.confidence)
            }
        }
    }
}
