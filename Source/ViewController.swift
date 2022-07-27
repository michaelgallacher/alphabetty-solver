//
//  ViewController.swift
//
//  Copyright © 2019 Michael Gallacher. All rights reserved.
//

import UIKit
import Foundation
import Photos

var wordTrie = Trie()

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var importButton: UIButton!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var wordListTableView: UITableView!
    @IBOutlet weak var imageViewOverlay: UIImageView!
    @IBOutlet weak var lastLoadTime: UILabel!
    @IBOutlet weak var resultBoardLabel: UILabel!
    
    private var word_list:[(String, [Tile])] = []
    private var original_word_list:[(String, [Tile])] = []
    
    private var letter_rects:[CGRect] = []
    private var boardDebugDescription: String = ""
    private var imageUrl = NSURL()
    private var currentBoard:Board? = nil
    private let numRows:Int = 8
    private let numCols:Int = 7
    
    func _init() {
        if let path = Bundle.main.path(forResource: "scowl_60_0_no_special", ofType: "txt") {
            do {
                let start = Date.timeIntervalSinceReferenceDate
                let data = try String(contentsOfFile: path, encoding: .utf8)
                Set<String>(data.components(separatedBy: .newlines)).forEach {wordTrie.insert($0)}
                let time = Date.timeIntervalSinceReferenceDate - start
                print("time to load: " + String(time))
            } catch {
                print(error)
            }
        }
    }
    
    /// Tanslates the given point in to the coordinate space of the actual image iteself.
    /// This is different that the coordinate space of the UIImageView since the image
    /// likely does not fit the image view perfectly.
    /// - Parameters:
    ///   - point: The point within the view
    ///   - view: The image view with an aspect-filled image
    /// - Returns: A point relative to the image.
    func aspectFillPoint(for point: CGPoint, in view: UIImageView) -> CGPoint {
        guard let img = view.image else {
            return CGPoint.zero
        }
        
        assert(view.contentMode == .scaleAspectFill)
        
        var imgSize = img.size
        let viewSize = view.frame.size
        
        let aspectWidth  = viewSize.width / imgSize.width
        let aspectHeight = viewSize.height / imgSize.height
        
        let f = max(aspectWidth, aspectHeight)
        imgSize.width *= f
        imgSize.height *= f
        
        let xOffset = (viewSize.width - imgSize.width) / 2.0
        let yOffset = (viewSize.height - imgSize.height) / 2.0
        
        return CGPoint(
            x: Int((point.x - xOffset) / f) ,
            y: Int((point.y - yOffset) / f) 
        )
    }
    
    
    var textField: UITextField  {
        let txt =  UITextField(frame: CGRect(x:0 , y:0, width:200, height: 60))
        txt.textColor = .white
        txt.backgroundColor = .black
        txt.placeholder = "new letter"
        txt.textAlignment = .left
        let ctr = self.view.center
        print(ctr.x)
        txt.center = ctr
        txt.translatesAutoresizingMaskIntoConstraints = true
        return txt
    }
    
    @objc
    func cancelAction() {
        self.textField.resignFirstResponder()
    }
    
    @objc
    func doneAction() {
        self.textField.resignFirstResponder()
    }
    
    @IBAction func onImageLongPressed(_ sender: Any) {
        guard let sender = sender as? UILongPressGestureRecognizer else { return }
        print(sender.state.rawValue)
        if sender.state == .began {
            self.view.addSubview(self.textField)
        }
    }
    
    @IBAction func onImageLongPressed2(_ sender: Any) {
        guard let sender = sender as? UITapGestureRecognizer else { return }
        guard let board = self.currentBoard else { return }
        
        // The image is presented as 'aspect fill' so calculate
        // where on the image the tap occured
        let pt = sender.location(in: imageView)
        let apt = aspectFillPoint(for: pt, in: imageView)
        
        var idx = 0
        for rect in letter_rects {
            if rect.contains(apt) {
                break
            }
            idx += 1
        }
        
        // If the index is valid, find all words that use the tile at that index.
        // Otherwise, clear the board.
        if idx < board.tiles.count {
            let selectedTile = board.tiles[idx];
            self.wordListTableView.reloadData()
        }
    }
    
    
    //
    // Determine which, if any, tile was tapped on and then display all words
    // which contain the letter in the tile.
    //
    @IBAction func onImageTapped(_ sender: Any) {
        guard let sender = sender as? UITapGestureRecognizer else { return }
        guard let board = self.currentBoard else { return }
        
        // The image is presented as 'aspect fill' so calculate
        // where on the image the tap occured
        let pt = sender.location(in: imageView)
        let apt = aspectFillPoint(for: pt, in: imageView)
        
        var idx = 0
        for rect in letter_rects {
            if rect.contains(apt) {
                break
            }
            idx += 1
        }
        
        var selectedTiles: [Tile] = []
        // If the index is valid, find all words that use the tile at that index.
        // Otherwise, clear the board.
        if idx < board.tiles.count {
            word_list = []
            let selectedTile = board.tiles[idx];
            selectedTiles.append(selectedTile!)
            for word in self.original_word_list {
                let tileList:[Tile] = word.1
                if tileList.contains(where: { $0.row == selectedTile!.row && $0.col == selectedTile!.col }) {
                    word_list.append(word)
                }
            }
            self.selectTiles(selectedTiles)
            self.wordListTableView.reloadData()
        } else {
            word_list = original_word_list
            self.selectTiles(selectedTiles)
            self.wordListTableView.reloadData()
        }
    }
    
    override public init(nibName nibNameOrNil: String?, bundle nibBundleOrNil: Bundle?) {
        super.init(nibName: nibNameOrNil, bundle: nibBundleOrNil)
        _init()
    }
    
    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
        _init()
    }
    
    func grokLetters(_ input: String) -> [(String,[Tile])] {
        self.currentBoard = Board(rowCount: self.numRows, colCount: self.numCols, contents: input)
        // get all the words + tiles
        let longWords = Array(self.currentBoard!.findAllWords().filter({$0.0.count >= 3}))
        let sortedLongWords = longWords.sorted(by: { ($0.0.count > $1.0.count) ? true : ( $0.0.count == $1.0.count ? $0.0 < $1.0 : false )})
        return sortedLongWords
    }
    
    func loadImage(imageUrl:NSURL) {
        self.imageView.image = UIImage(contentsOfFile: imageUrl.path!)
        self.imageUrl = imageUrl
        _updateBoard(imageUrl: imageUrl)
    }
    
    func loadImage(image:UIImage) {
        self.imageView.image = image
        self.imageUrl = NSURL()
        _updateBoard(image: image)
    }
    
    func _updateBoard(imageUrl:NSURL) {
        let start = Date.timeIntervalSinceReferenceDate
        let lettersAndRectsString = OpenCVWrapper.findLetters(with: imageUrl as URL, withRows: Int32(self.numRows), withCols: Int32(self.numCols))
        let findLetterTime = String(format: "%.2f", Date.timeIntervalSinceReferenceDate - start)
        print("Time for findletters: " + findLetterTime)
        __updateBoard(lettersAndRectsString)
        let totalTime = String(format: "%.2f", Date.timeIntervalSinceReferenceDate - start)
        self.lastLoadTime.text = totalTime + "/" + findLetterTime
    }
    
    func _updateBoard(image:UIImage) {
        let start = Date.timeIntervalSinceReferenceDate
        let lettersAndRectsString = OpenCVWrapper.findLetters(in: image, withRows: Int32(self.numRows), withCols: Int32(self.numCols))
        print("Time for findletters: " + String(Date.timeIntervalSinceReferenceDate - start))
        __updateBoard(lettersAndRectsString)
    }
    
    func __updateBoard(_ lettersAndRectsString:String) {
        self.letter_rects = []
        if (!lettersAndRectsString.contains(";") ||
            lettersAndRectsString.components(separatedBy: ";")[0].replacingOccurrences(of: "_", with: "").count == 0) {
            DispatchQueue.main.async {
                let alert = UIAlertController(title: "Not a recognized board", message: "Try a different image", preferredStyle: .alert)
                alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
                self.present(alert, animated: true)
            }
            self.original_word_list = []
        } else if (lettersAndRectsString[lettersAndRectsString.startIndex] == "!") {
            DispatchQueue.main.async {
                let alert = UIAlertController(title: "Exception?", message: lettersAndRectsString, preferredStyle: .alert)
                alert.addAction(UIAlertAction(title: "Yes", style: .default, handler: nil))
                self.present(alert, animated: true)
            }
            self.original_word_list = []
        } else {
            let rawComponents = lettersAndRectsString.components(separatedBy: ";")
            var clean = rawComponents[0]
            let rectStrings = rawComponents[1...]
            self.letter_rects = rectStrings.map() {
                let rc = $0.components(separatedBy:",")
                return CGRect(x: Int(rc[0])!, y: Int(rc[1])!, width: Int(rc[2])!, height: Int(rc[3])!)
            }
            
            clean.removeAll { $0 == "\n" }
            self.original_word_list = self.grokLetters(clean)     
            
            self.boardDebugDescription = ""
            var i = 0
            for c in rawComponents[0] {
                if i % numCols == 0 {
                    boardDebugDescription += "\n"
                }
                boardDebugDescription += " " + String(c)
                i += 1
            }
            self.resultBoardLabel.text = boardDebugDescription
            print(boardDebugDescription)
        }
        
        self.word_list = self.original_word_list
        self.selectTiles([])
        self.wordListTableView.reloadData()
        
        if (self.word_list.count > 0) {
            let indexPath = IndexPath(row:0, section:0)
            self.wordListTableView.selectRow(at: indexPath , animated: true, scrollPosition: .top)
            self.tableView(self.wordListTableView, didSelectRowAt: indexPath)
        }
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let pickedImage = info[UIImagePickerController.InfoKey.imageURL] as? NSURL {
            picker.dismiss(animated: true, completion: nil)
            DispatchQueue.main.async {
                self.loadImage(imageUrl: pickedImage)
            }
        }
    }
    
    @IBAction func onImportImage(_ sender: UIButton) {
        loadLastImage()
    }
    @IBAction func onImportImage2(_ sender: UIButton) {
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self
        imagePickerController.sourceType = UIImagePickerController.SourceType.savedPhotosAlbum
        imagePickerController.allowsEditing = false
        self.present(imagePickerController, animated: true, completion: nil)
    }
    
    func loadLastImage() {
        queryLastPhoto(resizeTo: nil) {
            image in
            guard let image = image else {return}
            self.loadImage(image: image)
        }
    }
    
    func queryLastPhoto(resizeTo size: CGSize?, queryCallback: @escaping ((UIImage?) -> Void)) {
        let fetchOptions = PHFetchOptions()
        fetchOptions.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        fetchOptions.includeAssetSourceTypes = .typeUserLibrary
        fetchOptions.fetchLimit = 1
        
        let fetchResult = PHAsset.fetchAssets(with: PHAssetMediaType.image, options: fetchOptions)
        if let asset = fetchResult.firstObject {
            
            let requestOptions = PHImageRequestOptions()
            requestOptions.isSynchronous = true
            requestOptions.deliveryMode = .highQualityFormat
            requestOptions.isNetworkAccessAllowed = true
            requestOptions.resizeMode = .none
            
            let manager = PHImageManager.default()
            manager.requestImage(for: asset,
                                 targetSize: PHImageManagerMaximumSize,
                                 contentMode: .aspectFit,
                                 options: requestOptions,
                                 resultHandler: { image, info in queryCallback(image) })
        }
    }
}

extension ViewController: UITableViewDelegate, UITableViewDataSource {
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return word_list.count == 0 || word_list[0].0.isEmpty ? 1 : word_list.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        if word_list.count == 0 || word_list[0].0.isEmpty || word_list[indexPath.item].0.count > 30 {
            let cell = tableView.dequeueReusableCell(withIdentifier: "debug_cell", for: indexPath as IndexPath) as UITableViewCell
            cell.textLabel?.text = "No words found!"
            return cell
        } else {
            let cell = tableView.dequeueReusableCell(withIdentifier: "word_cell", for: indexPath as IndexPath) as UITableViewCell
            cell.textLabel?.text = word_list[indexPath.item].0
            return cell
        }
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        if (word_list.count > 0) {
            let selectedWord = word_list[indexPath.row]
            selectTiles(selectedWord.1)
        } 
    } 
    
    func selectTiles(_ tiles: [Tile]) {
        let imageSize = self.imageView.image?.size
        let rect = CGRect(origin: CGPoint.zero, size: imageSize!)
        UIGraphicsBeginImageContext(imageSize!)
        UIGraphicsGetCurrentContext()!.clear(rect)
        UIGraphicsGetCurrentContext()!.setFillColor(red: 0, green: 0, blue: 0, alpha: 0.5)
        UIGraphicsGetCurrentContext()!.addRect(rect) 
        UIGraphicsGetCurrentContext()!.drawPath(using: .fill) 
        UIGraphicsGetCurrentContext()!.setFillColor(red: 255, green: 255, blue: 255, alpha: 0.0)
        
        for tile in tiles {
            let index = tile.row * numCols + tile.col
            let rect = letter_rects[Int(index)]
            UIGraphicsGetCurrentContext()!.clear(rect)
            UIGraphicsGetCurrentContext()!.addRect(rect) 
            UIGraphicsGetCurrentContext()!.drawPath(using: .fill) 
        }
        
        self.imageViewOverlay.frame = self.imageView.frame
        self.imageViewOverlay.image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
    }
} 


extension String {
    subscript (i: Int) -> Character {
        return self[index(startIndex, offsetBy: i)]
    }
    subscript (bounds: CountableRange<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[start ..< end]
    }
    subscript (bounds: CountableClosedRange<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[start ... end]
    }
    subscript (bounds: CountablePartialRangeFrom<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(endIndex, offsetBy: -1)
        return self[start ... end]
    }
    subscript (bounds: PartialRangeThrough<Int>) -> Substring {
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[startIndex ... end]
    }
    subscript (bounds: PartialRangeUpTo<Int>) -> Substring {
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[startIndex ..< end]
    }
}
extension Substring {
    subscript (i: Int) -> Character {
        return self[index(startIndex, offsetBy: i)]
    }
    subscript (bounds: CountableRange<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[start ..< end]
    }
    subscript (bounds: CountableClosedRange<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[start ... end]
    }
    subscript (bounds: CountablePartialRangeFrom<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(endIndex, offsetBy: -1)
        return self[start ... end]
    }
    subscript (bounds: PartialRangeThrough<Int>) -> Substring {
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[startIndex ... end]
    }
    subscript (bounds: PartialRangeUpTo<Int>) -> Substring {
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[startIndex ..< end]
    }
}
