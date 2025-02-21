import Foundation
import Combine
import Alamofire

final class APIClient {
    private let baseURL = URL(string: "http://localhost:8000/api/v1")!
    private let session: Session
    
    init() {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 300
        configuration.waitsForConnectivity = true
        
        session = Session(configuration: configuration)
    }
    
    func search(query: String) -> AnyPublisher<[Book], Error> {
        let request = SearchRequest(query: query)
        return performRequest(endpoint: "search", method: .post, parameters: request)
    }
    
    func getBook(id: Int) -> AnyPublisher<Book, Error> {
        return performRequest(endpoint: "books/\(id)", method: .get)
    }
    
    func query(question: String) -> AnyPublisher<QueryResponse, Error> {
        let request = QueryRequest(question: question)
        return performRequest(endpoint: "query", method: .post, parameters: request)
    }
    
    private func performRequest<T: Decodable, P: Encodable>(
        endpoint: String,
        method: HTTPMethod,
        parameters: P? = nil
    ) -> AnyPublisher<T, Error> {
        let url = baseURL.appendingPathComponent(endpoint)
        
        return session.request(url, method: method, parameters: parameters)
            .validate()
            .publishDecodable(type: T.self)
            .value()
            .mapError { $0 as Error }
            .eraseToAnyPublisher()
    }
}
