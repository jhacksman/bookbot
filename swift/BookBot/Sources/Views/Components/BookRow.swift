import SwiftUI

struct BookRow: View {
    let book: Book
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(book.title)
                .font(.headline)
            Text(book.author)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }
}
