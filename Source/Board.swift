//
//  Board.swift
//
//  Copyright © 2019 Michael Gallacher. All rights reserved.
//

import Foundation

/// Class that represents the collection of tiles on the playing surface
class Board : CustomStringConvertible {
    private(set) var rowCount:Int
    private(set) var colCount:Int
    private(set) var tiles = [Optional<Tile>]()
    
    /// Initializes the instace with a given row and column count.
    /// The board is populated in column-major order based on the
    /// value of 'contents'.
    ///
    /// The length of 'content' must equal rowCount * colCount.
    ///
    /// - Parameters:
    ///   - rowCount: The number of rows for this board.
    ///   - colCount: The number of colums for this board.
    ///   - contents: A column-major list of characters to initialize the board.
    init(rowCount:Int, colCount:Int, contents:String) {
        self.rowCount = -1
        self.colCount = -1
        self.tiles = []
        self.rowCount = rowCount
        self.colCount = colCount
        
        assert(contents.count == rowCount * colCount, "\(contents.count) \(rowCount * colCount)")
        
        // Create the tiles
        var iter = contents.makeIterator()
        for r in 0..<rowCount {
            for c in 0..<colCount {
                self.tiles.append(Tile(letter: Letter(iter.next()!), row: r, col: c))
            }
        }
        
        // Create a cached list of neighbors for each tile.
        for r in 0..<rowCount {
            for c in 0..<colCount {
                if let curTile = self.at(r,c) {
                    curTile.neighbors = self.getNeighbors(row: r,col: c)
                }
            }
        }
    }
    
    public var description: String { return "Board: \(tiles)" }
    
    /// Helper function to do bounds-checking
    /// - Parameters:
    ///   - row: The zero-based index of the row to retrieve.
    ///   - col: The zero-based index of the column to retrieve
    /// - Returns: If the position is value, the tile at that position; otherwise nil.
    private func at(_ row:Int, _ col:Int) -> Optional<Tile> {
        if row < 0 || row >= rowCount || col < 0 || col >= colCount {
            return nil
        }
        return tiles[row * colCount + col]
    }
    
    /// Helper to return the collection of tiles in all eight directions
    /// around the given tile specified by (row,col)
    /// - Parameters:
    ///   - row: The zero-based index of the row to retrieve.
    ///   - col: The zero-based index of the column to retrieve
    /// - Returns: An array of tiles starting in the upper-left corner
    private func getNeighbors(row:Int, col:Int) -> [Tile] {
        var neighbors:[Tile] = []
        let iter = [-1, 0, 1]
        for rowOffset in iter {
            for colOffset in iter {
                if rowOffset == 0 && colOffset == 0 { continue }
                if let n = self.at(row + rowOffset, col + colOffset) {
                    neighbors.append(n)
                }
            }
        }
        return neighbors
    }
    
    /// Returns all valid words which are reachable starting with the given tile
    /// - Parameters:
    ///   - curTile: The initial tile from which to begin the search.
    ///   - visitedTiles: A list of tiles that tiles that should not be explored
    ///   - results: A list of all words found, as specified by the tuple:(<word>, [tiles])
    private func allWords(curTile:Tile, visitedTiles:[Tile], results:inout [(String, [Tile])]) {
        if curTile.letter == "*" {
            let atoz = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for c in atoz {
                let starCandidate = Tile(letter: String(c), row: Int(curTile.row), col: Int(curTile.col))
                allWords(curTile: starCandidate, visitedTiles: visitedTiles, results: &results)
            }
            return
        }
        
        var curVisitedTiles = visitedTiles
        curVisitedTiles.append(curTile)
        
        let prefix = curVisitedTiles.map {$0.letter}
        if wordTrie.findLastNodeOf(prefix) == nil {
            return
        }
        
        for newNeighbor in curTile.neighbors.filter({!curVisitedTiles.contains($0)}) {
            allWords(curTile: newNeighbor, visitedTiles: curVisitedTiles, results: &results)
        }
        
        let candidateWord = prefix.joined()
        if wordTrie.containsWord(prefix) {
            results.append((candidateWord, [Tile](curVisitedTiles)))
        }
    }
    
    /// Returns all valid words which are reachable starting with the given tile
    /// - Returns: A list of all words found, as specified by the tuple:(<word>, [tiles])
    func findAllWords() -> [(String,[Tile])] {
        var words:[(String,[Tile])] = []
        for tile in tiles {
            if let la = tile {
                var combos:[(String,[Tile])] = []
                allWords(curTile: la, visitedTiles: [], results: &combos)
                words.append(contentsOf: combos)
            }
        }
        return words
    }
}
