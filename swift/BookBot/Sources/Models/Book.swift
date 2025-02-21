import Foundation

struct Book: Identifiable, Codable {
    let id: Int
    let title: String
    let author: String
    let description: String?
    let contentHash: String
    let metadata: BookMetadata
    let vectorId: String
    
    struct BookMetadata: Codable {
        let publicationDate: String?
        let publisher: String?
        let isbn: String?
        let language: String
        let format: String?
        let tags: [String]
        let series: String?
        let seriesIndex: Double?
    }
}
