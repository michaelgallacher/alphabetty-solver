//  Trie.swift

import Foundation

/// A node in the trie
class TrieNode<T: Hashable> {
    var value: T?
    weak var parentNode: TrieNode?
    var children: [T: TrieNode] = [:]
    var isTerminating = false
    var isLeaf: Bool {
        return children.count == 0
    }
    
    /// Initializes a node.
    /// - Parameters:
    ///   - value: The value that goes into the node
    ///   - parentNode: A reference to this node's parent
    init(value: T? = nil, parentNode: TrieNode? = nil) {
        self.value = value
        self.parentNode = parentNode
    }
    
    /// Adds a child node to self.  If the child is already present, do nothing.
    /// - Parameter value: The item to be added to this node.
    func add(value: T) {
        guard children[value] == nil else {
            return
        }
        children[value] = TrieNode(value: value, parentNode: self)
    }
}

/// A trie data structure containing words.  Each node is a single character of a word.
class Trie: NSObject {
    typealias Node = TrieNode<String>
    /// The number of words in the trie
    public var count: Int {
        return wordCount
    }
    /// Is the trie empty?
    public var isEmpty: Bool {
        return wordCount == 0
    }
    fileprivate let root: Node
    fileprivate var wordCount: Int
    
    /// Creates an empty trie.
    override init() {
        root = Node()
        wordCount = 0
        super.init()
    }
    
    /// Inserts a word into the trie.  If the word is already present,
    /// there is no change.
    ///
    /// Assumes that 'q' is always followed by 'u'.
    ///
    /// - Parameter word: the word to be inserted.
    func insert(_ word: String) {
        var input:[Letter] = []
        var skipNext = false;
        for character in word {
            if (!skipNext) {
                if (character == "Q") {
                    input.append("QU")
                    skipNext = true;
                } else {
                    input.append(String(character))
                }
            } else {
                skipNext = false
            }
        }
        insert(input)
    }
    
    /// Insert the given word into the trie.
    /// - Parameter word: A word, represented as a list of letters
    private func insert(_ word: [Letter]) {
        guard !word.isEmpty else {
            return
        }
        var currentNode = root
        for character in word {
            if let childNode = currentNode.children[character] {
                currentNode = childNode
            } else {
                currentNode.add(value: character)
                currentNode = currentNode.children[character]!
            }
        }
        
        // Word already present?
        guard !currentNode.isTerminating else {
            return
        }
        wordCount += 1
        currentNode.isTerminating = true
    }
}

extension Trie {
    /// Determines whether a word is in the trie.
    /// - Parameter word: the word to check for
    /// - Returns: true if the full word is present, 
    ///            false if the prefix and word are not present.
    func containsWord(_ word: [Letter]) -> Bool {
        return findLastNodeOf(word)?.isTerminating ?? false
    }
    
    
    /// Finds the last node of the given prefix
    /// - Parameter prefix: A string representing a word or a prefix thereof.
    /// - Returns: A Node if the entire string is found in the trie.  Otherwise, nil.
    func findLastNodeOf(_ prefix: [Letter]) -> Node? {
        assert(!prefix.isEmpty, "prefix/word should not be empty")
        var currentNode = root
        for character in prefix {
            guard let childNode = currentNode.children[character] else {
                return nil
            }
            currentNode = childNode
        }
        return currentNode
    }
}
