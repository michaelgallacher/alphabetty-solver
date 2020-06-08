//
//  OpenCVWrapper.h
//
//  Copyright © 2019 Michael Gallacher. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface OpenCVWrapper : NSObject

+ (NSString*) findLettersInImage: (UIImage*) inputImage
                        withRows: (int) rows 
                        withCols: (int) cols;
+ (NSString*) findLettersWithUrl: (NSURL*) inputUrl 
                        withRows: (int) rows 
                        withCols: (int) cols;

@end

NS_ASSUME_NONNULL_END
