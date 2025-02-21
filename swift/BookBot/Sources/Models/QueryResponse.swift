import Foundation

struct QueryResponse: Codable {
    let answer: String
    let citations: [Citation]
    let confidence: Double
    
    struct Citation: Identifiable, Codable {
        let id: String
        let bookId: String
        let title: String
        let author: String
        let quotedText: String
        let relevanceScore: Double
    }
}
