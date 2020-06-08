//
//  Tile.swift
//
//  Copyright © 2019 Michael Gallacher. All rights reserved.
//

import Foundation

typealias Letter = String

class Tile : CustomStringConvertible, Equatable {
    static private var idCounter:Int = 0
    private var id:Int = 0
    private(set) var letter:Letter
    private(set) var row:Int
    private(set) var col:Int
    var neighbors:[Tile]

    @inlinable
    static func == (lhs: Tile, rhs: Tile) -> Bool {
        if lhs.col == rhs.col && lhs.row == rhs.row {
            assert(lhs.id == rhs.id)
            return true
        }
        return false
    }
    
    init(letter:Letter, row:Int, col:Int) {
        self.letter = letter == "Q" ? "QU" : letter
        self.neighbors = []
        self.id = Tile.idCounter
        self.row = row
        self.col = col
        Tile.idCounter += 1
    }
  
    public var description: String { return "Letter: \(letter)(\(self.id));" }
}
