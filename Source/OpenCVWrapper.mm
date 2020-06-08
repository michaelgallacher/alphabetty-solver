//
//  OpenCVWrapper.m
//
//  Copyright © 2019 Michael Gallacher Gallacher. All rights reserved.
//

#import <vector>
#import <numeric>
#import <opencv2/opencv.hpp>
#import <opencv2/imgproc.hpp>
#import <opencv2/imgcodecs/ios.h>

#import <Foundation/Foundation.h>
#import "OpenCVWrapper.h"

using namespace std;

struct Thresh {
    Thresh(int channel, int value) {
        this->channel = channel;
        this->value = value;
    }
    int channel;
    int value;
};

const int g_debug_level = -2;

const int g_fs_gutter = 2;

const int g_find_letter_morph = 3;
const int g_match_morph = 3;
const int g_fbm_min_crop = 5;
const int g_fbm_max_crop = 15;
const int g_fbm_crop_step = 10;
const cv::Size g_gaussian_blur(3, 3);

//
// A collection of thresholds to consider when evaluating a tile.
// Except for the 'normal' threshold, the thresholds are based on
// board in which tiles are particularly difficult to identify.
const auto NORM_TILE_THRESH = Thresh(2, -168);
const auto NORM_TILE_THRESH2 = Thresh(2, -136);
const auto R_ON_Y_TILE_THRESH = Thresh(1, -168);
const auto Y_ON_B_TILE_THRESH = Thresh(0, -140);
const auto Y_ON_B_TILE_THRESH2 = Thresh(2, 188);
const auto Y_ON_PU_TILE_THRESH = Thresh(1, 200);
const auto VINES = Thresh(1, -114);
const auto BUBBLES = Thresh(2, -180);
const Thresh THRESHOLDS[] = {NORM_TILE_THRESH, R_ON_Y_TILE_THRESH, Y_ON_B_TILE_THRESH, Y_ON_B_TILE_THRESH2, Y_ON_PU_TILE_THRESH, VINES, BUBBLES, NORM_TILE_THRESH2};


static void f_print(NSString* out, int level) {
    if (g_debug_level == -1 || level <= g_debug_level){
        NSLog(@"%@", out);
    }
}

static void DebugOutput(NSString* out, int level = 2) {
    if (level <= g_debug_level) {
        NSLog(@"%@", out);
    }
}

static void printParams() {
    DebugOutput([NSString stringWithFormat: @"g_post_sobel_morph: %d", g_find_letter_morph], 1);
    DebugOutput([NSString stringWithFormat: @"g_match_morph: %d", g_match_morph], 1);
    DebugOutput([NSString stringWithFormat: @"g_fbm_min_crop: %d", g_fbm_min_crop], 1);
    DebugOutput([NSString stringWithFormat: @"g_fbm_max_crop: %d", g_fbm_max_crop], 1);
}  


// Diagnostic function to print an ASCII representation of the image.
static void printImg(cv::Mat& input) {
    int width = input.size[1];
    uchar* pdata = input.data;
    NSString* s = @"";
    int col = 0;
    for(uchar* pc = pdata; pc < input.dataend; pc++) {
        s = [s stringByAppendingString:[NSString stringWithFormat:@"%d ", *pc]];
        if (++col == width) {
            col = 0;
            s = [s stringByAppendingString:@"\n"];
        }
    }
    DebugOutput(s);
}


static void sobel(cv::Mat& input, cv::Mat& output) {
    cv::Mat blur;
    cv::GaussianBlur(input, blur, cv::Size(3, 3), 0);
    
    cv::Mat grad_x;
    cv::Sobel(blur, grad_x, CV_16S, 1, 0);
    cv::Mat grad_y;
    cv::Sobel(blur, grad_y, CV_16S, 0, 1);
    
    cv::Mat abs_grad_x;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::Mat abs_grad_y;
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    cv::addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0, output);
}

typedef vector<cv::Point> Contour;

static bool compareInterval(Contour contour1, Contour contour2) { 
    auto area1 = cv::contourArea(contour1);
    auto area2 = cv::contourArea(contour2);
    return  area1 > area2;
} 

static bool calcMomentDistance(Contour& biggest_contour, tuple<int,int>& result) {
    auto moment = cv::moments(biggest_contour);
    auto divisor = moment.m00;
    if (divisor == 0) {
        f_print(@"letter rejected due to null moment", 2);
        return false;
    }
    
    auto cX = int(moment.m10 / divisor);
    auto cY = int(moment.m01 / divisor);
    result = tuple<int,int>(cX, cY);
    return true;
}

static bool find_letter(cv::Mat& tile_img, cv::Mat& letter_img, int threshold, bool is_baseline_shape) {
    auto width = tile_img.size().width;
    auto height = tile_img.size().height;
    f_print([NSString stringWithFormat: @"fl-ts: %d, %d", width, height], 2);
    
    cv::Mat thresh_image;
    if (threshold > 0) {
        cv::threshold(tile_img, thresh_image, threshold, 255, cv::THRESH_BINARY);
    } else if (threshold < 0) {
        cv::threshold(tile_img, thresh_image, -threshold, 255, cv::THRESH_BINARY_INV);
    } else {
        cv::threshold(tile_img, thresh_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }
    
    // Start by finding contours that might be candidates to be a letter.
    cv::Mat img1;
    auto element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(g_find_letter_morph, g_find_letter_morph));
    cv::morphologyEx(thresh_image, img1, cv::MORPH_OPEN, element);
    
    vector<Contour> contours;
    vector<cv::Vec4i> heirarchy;
    cv::findContours(img1, contours, heirarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.size() == 0) {
        return false;
    }
    
    //
    //    if (!is_baseline_shape) {
    //        printImg(thresh_image);
    //    }
    //
    sort(contours.begin(), contours.end(), compareInterval);
    
    int index_of_biggest = 0;
    Contour biggest_contour;
    biggest_contour = contours[index_of_biggest];
    
    // calc moment
    tuple<int,int> moment;
    if (!calcMomentDistance(biggest_contour, moment)) {
        return false;
    }
    // Letters have moments within a certain range of the center of the tile.
    auto dist = sqrt(pow(get<0>(moment)-width / 2, 2) + pow(get<1>(moment)-height / 2, 2));
    auto max_moment = width * height / 200;
    if (dist > max_moment) {
        return false;
    }
    
    auto br = cv::boundingRect(biggest_contour);
    f_print([NSString stringWithFormat: @"br: (%d, %d, %d, %d)", br.tl().x, br.tl().y, br.width, br.height], 2);
    
    // If the bounding rect of the candidate letter takes up the entire tile, it's not a letter.
    if (br.area() > width * height * 0.95) {
        return false;
    }
    // If the bounding rect is too small or the ratio is off, it's not a letter.
    if (br.width < width * 0.1 || br.height < height * 0.4) {
        return false;
    }
    
    letter_img = img1(br);
    
    f_print(@"found letter", 2);
    return true;
}

// Compare an image against a basline letter to see if we consider them the same.
static double cvmatchShapes(cv::Mat& candidate, cv::Mat& baseline) {
    auto candidate_w = int(candidate.size[1]);
    auto candidate_h = int(candidate.size[0]);
    auto baseline_w = int(baseline.size[1]);
    auto baseline_h = int(baseline.size[0]);
    
    f_print([NSString stringWithFormat: @"cvmatch_shapes w,h: %d, %d; %d, %d", candidate_w, candidate_h, baseline_w, baseline_h], 2);
    
    // Use double to mirror Python
    double r1 = double(candidate_h) / candidate_w;
    double r2 = double(baseline_h) / baseline_w;
    double r_dim = abs(r2/r1);
    if (r_dim > 1) {
        r_dim = abs(r1/r2);
    }
    if (r_dim < 0.8) {
        f_print([NSString stringWithFormat: @"cvm: rejected due to dimension ratio %.3f", r_dim], 2);
        return -1;
    }
    
    auto element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(g_match_morph, g_match_morph));
    cv::Mat candidateMorph;
    cv::morphologyEx(candidate, candidateMorph, cv::MORPH_DILATE, element);
    
    // Adjust the images to be the same size.
    //    cv::Mat mat1_resized;
    //    cv::resize(candidate, mat1_resized, cv::Size(m2w, m2h));
    cv::Mat mat2_resized;
    auto _interpolation = baseline_h < candidate_h ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(baseline, mat2_resized, cv::Size(candidate_w, candidate_h), _interpolation);
    
    cv::Mat diff_img;
    cv::bitwise_xor(candidateMorph, mat2_resized, diff_img);
    
    cv::Mat diff_img_morph;
    cv::morphologyEx(diff_img, diff_img_morph, cv::MORPH_OPEN, element);
    
    cv::Mat diff_img_morph_thresh;
    cv::threshold(diff_img_morph, diff_img_morph_thresh, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
    
    double diff = (double)cv::countNonZero(diff_img_morph_thresh) / (candidate_w * candidate_h);
    f_print([NSString stringWithFormat: @"cvm: match with diff: %.3f, r_dim: %.3f", diff, r_dim], 2);
    return diff;
}

static void find_squares(cv::Mat& gray, vector<cv::Rect>& bounding_rects, vector<Contour>& sorted_contours, int thresh, bool single_tile=false) {
    double min_area = (double)gray.size[0] * gray.size[1] / 200.0;
    
    cv::Mat bin1;
    sobel(gray, bin1);
    
    cv::Mat bin2;
    if (thresh > 0) {
        cv::threshold(bin1, bin2, thresh, 255, cv::THRESH_BINARY);
    } else if (thresh < 0) {
        cv::threshold(bin1, bin2, -thresh, 255, cv::THRESH_BINARY_INV);
    } else {
        if (single_tile) {
            cv::threshold(bin1, bin2, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else {
            cv::Mat tmp;
            auto otsu = cv::threshold(bin1, tmp, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            cv::threshold(bin1, bin2, otsu/2, 255, cv::THRESH_BINARY);
        }
    }
    
    auto element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::Mat diff_img_morph;
    cv::morphologyEx(bin2, diff_img_morph, cv::MORPH_ERODE, element);
    
    cv::findContours(diff_img_morph, sorted_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    sort(sorted_contours.begin(), sorted_contours.end(), compareInterval);
    
    for (auto orig_contour : sorted_contours) {
        auto area = cv::contourArea(orig_contour);
        if (single_tile || ( (min_area < area && area < 5 * min_area))) {
            auto br = cv::boundingRect(orig_contour);
            double r = (double)br.width / br.height;
            r = r > 1.0 ?  (double)br.height / br.width : r;
            if (r > 0.95){
                double bra = br.area();
                if (area > bra * 0.9) {
                    bounding_rects.push_back(br);
                }
            }
        }
    }
}

// Find squares based on a best-guess location of the board.  Hacky and shouldn't ever be needed.
static void find_squares_ab(cv::Mat& gray, vector<cv::Rect>& letters, int num_row_tiles, int num_col_tiles) {
    letters.clear();
    
    int img_width = gray.size[1];
    int img_height = gray.size[0];
    
    if (img_height > img_width) {
        // portrait
        const int top_offset = int(round(img_height / 45));
        const int screen_edge = int(round(img_width / 200));
        const int blue_edge = int(round(img_width / 50));
        const int tile_gaps = int(round(img_width / 75));
        const int edge = screen_edge + blue_edge;
        
        const int tile_width = int((img_width - (2 * edge) - ((num_col_tiles - 1) * tile_gaps)) / num_col_tiles);
        int letter_area_height = num_row_tiles * tile_width + (num_row_tiles - 1) * tile_gaps + (2 * blue_edge);
        
        const int top = int((img_height - letter_area_height) * 0.5) + top_offset;
        const int left = edge;
        
        DebugOutput([NSString stringWithFormat:@"size: %d, %d", num_col_tiles, num_row_tiles], 1);
        DebugOutput([NSString stringWithFormat:@"tile width: %d", tile_width], 1);
        DebugOutput([NSString stringWithFormat:@"image size: %d, %d", img_width, img_height], 1);
        DebugOutput([NSString stringWithFormat:@"top_offset: %d", top_offset], 1);
        DebugOutput([NSString stringWithFormat:@"blue_edge: %d", blue_edge], 1);
        DebugOutput([NSString stringWithFormat:@"screen_edge: %d", screen_edge], 1);
        DebugOutput([NSString stringWithFormat:@"tile_gaps: %d", tile_gaps], 1);
        DebugOutput([NSString stringWithFormat:@"letter_area_height: %d", letter_area_height], 1);
        DebugOutput([NSString stringWithFormat:@"left: %d", left], 1);
        DebugOutput([NSString stringWithFormat:@"top: %d", top], 1);
        
        for (int r = 0; r < num_row_tiles; r++) {
            for (int c = 0; c < num_col_tiles; c++) {
                cv::Rect2i letter_bounds(
                                         left + c * (tile_width + tile_gaps) - g_fs_gutter,
                                         top + blue_edge + r * (tile_width + tile_gaps) - g_fs_gutter,
                                         tile_width + 2 * g_fs_gutter,
                                         tile_width + 2 * g_fs_gutter);
                letters.push_back(letter_bounds);
                DebugOutput([NSString stringWithFormat: @"fs: (%d, %d, %d, %d)", letter_bounds.tl().x, letter_bounds.tl().y, letter_bounds.width, letter_bounds.height]);
            }
        }
    }
}

static tuple<int, double> _find_best_match(cv::Mat& target_letter, vector<tuple<string,cv::Mat>>& candidates) {
    double best_distance = 999999;
    int best_candidate_index = -1;
    
    if (target_letter.size().area() > 0) {
        // Try each candidate letter and keep the best match to the target.
        for (int candidateIndex = 0; candidateIndex < candidates.size(); candidateIndex++) {
            auto candidateLetter = candidates[candidateIndex];
            
            // Call OpenCV to determine the correlation between the letters.
            // print(f"target_letter: {target_letter}")
            // print(f"candidateLetter:{candidateLetter}")
            double distance = cvmatchShapes(target_letter, get<1>(candidateLetter));
            
            // The smaller the value ('distance'), the better the match.
            // Letters are in priority low to high; take the last one.
            if (distance >= 0 && distance <= best_distance) {
                // DebugOutput([NSString stringWithFormat: @"_fbm->letter: %s dist: %d count:%lu", get<0>(candidateLetter), distance, get<1>(candidateLetter).size()]);
                
                best_distance = distance;
                best_candidate_index = candidateIndex;
            }
            
            if (best_distance == 0) {
                break;
            }
        }
    }
    
    return tuple<int, double>(best_candidate_index, best_distance);
}

static tuple<int, int, double> find_best_match(cv::Mat& _src_img, vector<tuple<string,cv::Mat>>& candidates) {
    double best_distance = 999999;
    auto best_candidate_index = -1;
    auto best_letter_area = 0;
    
    auto img_width = _src_img.size[1];
    auto img_height = _src_img.size[0];
    
    cv::Mat src_img;
    cv::GaussianBlur(_src_img, src_img, g_gaussian_blur, 0);
    
    // Iterate over different cropping sizes of the outer edges of the square.
    for (int crop = g_fbm_min_crop; crop <= g_fbm_max_crop; crop += g_fbm_crop_step) {
        cv::Rect tile_rect(crop, crop, img_width - 2 * crop, img_height - 2 * crop);
        cv::Mat tile_image = src_img(tile_rect);
        cv::Mat channels[3];
        cv::split(tile_image, channels);
        // Iterate over all the possible board types, if necessary.
        for (auto threshold : THRESHOLDS) {
            cv::Mat target_letter;
            find_letter(channels[threshold.channel], target_letter, threshold.value, false);
            auto best_match = _find_best_match(target_letter, candidates);
            auto tmp_best_candidate_index = get<0>(best_match);
            double tmp_best_distance = get<1>(best_match);
            
            f_print([NSString stringWithFormat: @"fbm->i: %d d: %.3f", tmp_best_candidate_index, tmp_best_distance], 3);
            
            // If we found a new best, save info about the letter and keep going.
            if (tmp_best_distance >= 0 && tmp_best_distance <= best_distance) {
                best_distance = tmp_best_distance;
                best_candidate_index = tmp_best_candidate_index;
                best_letter_area = target_letter.size[0] * target_letter.size[1];
            }
            
            if (best_distance == 0) {
                break;
            }
        }
    }
    return make_tuple(best_letter_area, best_candidate_index, best_distance);
}

static vector<tuple<string,cv::Mat>> alphabet_baseline_contours;

static void create_ab_font_baseline() {
    
    if(alphabet_baseline_contours.size() > 0) {
        return;
    }
    // Sorting by frequency speeds up the brute-force search
    NSString* alphabet = @"ETAOINSRHLDCUMFPGWYBVKXJQZ";
//    NSString* alphabet = @"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    
    for (int i = 0; i < alphabet.length; i++) {
        NSString* letter = [NSString stringWithFormat:@"%c", [alphabet characterAtIndex:i]];
        NSString* fontpath = [NSString stringWithFormat:@"./resources_ab/%c", [alphabet characterAtIndex:i]];
        NSString* filePath = [[NSBundle mainBundle] pathForResource:fontpath ofType:@"png"];
        cv::Mat orig_img = cv::imread([filePath UTF8String]);
        
        cv::Mat letter_img;
        cv::GaussianBlur(orig_img, letter_img, g_gaussian_blur, 0);
        
        cv::Mat channels[3];
        cv::split(letter_img, channels);
        
        cv::Mat letter_img3;
        if (find_letter(channels[NORM_TILE_THRESH.channel], letter_img3, NORM_TILE_THRESH.value, true)) {
            alphabet_baseline_contours.push_back(make_tuple([letter UTF8String], letter_img3));
        } else {
            assert(false);
        }
    }
}

static bool has_blank_center_only(const cv::Mat& color_tile) {
    auto w = color_tile.size[1];
    auto h = color_tile.size[0];
    
    // ignore tiles which have no contours at all.
    cv::Mat gray_tile;
    cv::cvtColor(color_tile, gray_tile, cv::COLOR_BGR2GRAY);
    cv::Mat sobel_tile;
    sobel(gray_tile, sobel_tile);
    cv::Mat ranged;
    cv::inRange(sobel_tile, 64, 192, ranged);
    auto nz = cv::countNonZero(ranged);
    if (nz < 10) {
        DebugOutput(@"hbco: fail 1", 2);
        return false;
    }
    
    // mask out the inner square to check if the outer edge has
    // POI everywhere
    cv::Mat blank_middle = sobel_tile.clone();
    auto edge = int(h / 8);
    cv::Size edge_size(w-2*edge, h-2*edge);
    cv::Mat zeroes = cv::Mat::zeros(edge_size, CV_8UC1);
    zeroes.copyTo(blank_middle(cv::Rect(cv::Point(edge, edge), edge_size)));
    cv::Mat threshold_img;
    cv::threshold(blank_middle, threshold_img, 32, 255, cv::THRESH_BINARY);
    
    // close and count the remaining pixels
    auto element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat morphed;
    cv::morphologyEx(threshold_img, morphed, cv::MORPH_CLOSE, element);
    nz = cv::countNonZero(morphed);
    // If there are POI around the edges, the tile can't be blank.
    if (nz < h * w * 0.33) {
        DebugOutput(@"hbco: fail 2", 2);
        return false;
    }
    
    // Now look at the middle of the tile and see if there are any POI.
    auto crop = int(h / 3);
    cv::Size crop_size(w-2*crop, h-2*crop);
    cv::Mat cropped = sobel_tile(cv::Rect(cv::Point(crop, crop), crop_size));
    cv::Mat cropped_ranged;
    cv::inRange(cropped, 32, 192, cropped_ranged);
    nz = cv::countNonZero(cropped_ranged);
    
    if (nz < 10) {
        DebugOutput(@"hbco: pass 3", 2);
    }
    else {
        DebugOutput(@"hbco: fail", 2);
    }
    return nz < 10;
}


static bool has_squares_and_zero(const cv::Mat &color_tile) {
    vector<cv::Mat> _candidate_images;
    cv::split(color_tile, _candidate_images);
    reverse(_candidate_images.begin(), _candidate_images.end());
    
    for(cv::Mat candidate_img: _candidate_images) {
        auto ci_w = candidate_img.size[1];
        auto ci_h = candidate_img.size[0];
        for (int thresh=0; thresh < 256; thresh += 32) {
            vector<cv::Rect> squares;
            vector<Contour> contours;
            find_squares(candidate_img, squares, contours, thresh, true);
            DebugOutput(@"hsaz: pre-squares", 3);
            if (squares.size() >= 2 && squares.size() <= 4 && contours.size() > 2) {
                auto ca1 = cv::contourArea(contours[0]);
                auto ca2 = cv::contourArea(contours[1]);
                tuple<int,int> moment;
                if (!calcMomentDistance(contours[2], moment)) {
                    continue;
                }
                auto moment_dist = sqrt(pow(ci_w * .76 - get<0>(moment), 2) + pow(ci_h * .83  - get<0>(moment), 2));
                DebugOutput(@"hsaz: pre-moment", 3);
                if (moment_dist < 5) {
                    auto ba = ci_w * ci_h;
                    auto outer_ratio = ca1 / ba;
                    auto inner_ratio = ca2 / ca1;
                    DebugOutput(@"hsaz: pre-ratio", 3);
                    if (outer_ratio > 0.75 and inner_ratio > 0.75) {
                        DebugOutput(@"hsaz: found blank", 3);
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
}

static bool is_blank_tile(const cv::Mat& color_tile) {
    return has_blank_center_only(color_tile);

// TODO: figure out has_squares_and_zero. it's extremely
//       expensive and doesn't seem to have a positive hit.
//    return has_squares_and_zero(color_tile);
//    if (has_blank_center_only(color_tile)) {
//        return true;
//    }
//    return has_squares_and_zero(color_tile);
}


static void get_average_cluster_vals(const vector<int>& values, vector<int>& averages) {
    vector<vector<int>> clusters;
    for (auto value : values) {
        bool found_group = false;
        for (auto cluster_list : clusters) {
            auto dist = sqrt(pow(value - cluster_list[0], 2));
            if (dist < 30) {
                cluster_list.push_back(value);
                found_group = true;
                break;
            }
        }
        if (!found_group) {
            vector<int> new_cluster;
            new_cluster.push_back(value);
            clusters.push_back(new_cluster);
        }
    }
    
    for(auto cluster_list : clusters) {
        auto total = std::accumulate(cluster_list.begin(), cluster_list.end(), 0);
        averages.push_back(round((double)total/cluster_list.size()));
    }
    
    sort(averages.begin(), averages.end());
}

static void get_tile_size(vector<cv::Rect>& all_squares, int img_w, int img_h, int& tile_size, cv::Rect& tile_region){
    // Find centers
    vector<int> x_centers;
    vector<int> y_centers;
    for (auto sq : all_squares) {
        int cx = round(sq.tl().x + sq.width / 2.0);
        int cy = round(sq.tl().y + sq.height / 2.0);
        x_centers.push_back(cx);
        y_centers.push_back(cy);
    }
    
    // Find clusters
    vector<int> x_avgs;
    get_average_cluster_vals(x_centers, x_avgs);
    vector<int> x_diffs(x_avgs.size());
    adjacent_difference(x_avgs.begin(), x_avgs.end(), x_diffs.begin());
    
    vector<int> y_avgs;
    get_average_cluster_vals(y_centers, y_avgs);
    vector<int> y_diffs(y_avgs.size());
    adjacent_difference(y_avgs.begin(), y_avgs.end(), y_diffs.begin());
    
    // Remove the smallest value if it is unique.
    if(x_diffs.size() >= 3) {
        x_diffs.erase(x_diffs.begin());
    }
    
    map<int, size_t> count_map;
    for (auto v : x_diffs)
        ++count_map[v];
    for (auto v : y_diffs)
        ++count_map[v];
    
    vector<pair<int, size_t>> counts;
    for(auto kv: count_map) {
        counts.push_back(kv);
    }
    sort(counts.begin(), counts.end(), [] (const pair<int,size_t> &a, const pair<int,size_t> &b) {  return (a.second > b.second); });
    
    tile_size = counts[0].first;
    
    auto blue_edge = int(round(img_w / 50.0));
    auto top_offset = int(round(img_h / 45.0));
    auto screen_edge = int(round(img_w / 200.0));
    auto edge = blue_edge + screen_edge;
    
    auto max_tile_section_height = min(tile_size * 8 + blue_edge, int(round(img_h * 2.0 / 3)));
    auto top_bounds = round((img_h - max_tile_section_height) * 0.5) + top_offset;
    f_print([NSString stringWithFormat: @"tile diff: %d", tile_size], 3);
    
    auto tile_top = round(y_avgs[0] - tile_size / 2.0);
    while (tile_top - tile_size >= top_bounds - 1) {
        tile_top -= tile_size;
    }
    
    auto left_bounds = edge / 2.0;  // int(floor((img_w - (tile_size * max_cols)) / 2))
    auto tile_left = round(x_avgs[0] - tile_size / 2.0);
    while (tile_left - tile_size >= left_bounds - 1) {
        tile_left -= tile_size;
    }
    
    auto width = int(img_w - 2 * tile_left);
    auto height = int(top_bounds - tile_top + max_tile_section_height);
    
    tile_region = cv::Rect(int(tile_left), int(tile_top), width, height);
}

static void find_all_squares(const cv::Mat& src_img, vector<cv::Rect>& new_squares, const int n_rows, const int n_cols) {
    auto scale = src_img.size[1] < 800 ? 2 : 1;
    cv::Mat src_gray_tmp;
    cv::cvtColor(src_img, src_gray_tmp, cv::COLOR_BGR2GRAY);
    
    cv::Mat src_gray;
    if (scale == 1) {
        src_gray = src_gray_tmp;
    } else {
        cv::resize(src_gray_tmp, src_gray, cv::Size(0, 0), scale, scale);
    }
    
    vector<cv::Rect> all_squares;
    vector<Contour> contours;
    find_squares(src_gray, all_squares, contours, 0, false);
    auto img_h = src_gray.size[0];
    auto img_w = src_gray.size[1];
    
    if (all_squares.size() <= 1) {
        return;
    }
    
    int tile_size;
    cv::Rect tile_region;
    get_tile_size(all_squares, img_w, img_h, tile_size, tile_region);
    tile_size = round(tile_size / scale);
    int left = round(tile_region.tl().x / scale);
    auto top = round(tile_region.tl().y / scale);
    auto right = round(tile_region.br().x / scale);
    auto bottom = round(tile_region.br().y / scale);
    
    int rows = 0;
    int buffer = 2;
    // subtract just in case there were rounding issues.
    for (int y = top; y < bottom - 0 - 10; y += tile_size) {
        rows++;
        for (int x = left; x < right - 0 - 10; x += tile_size) {
            f_print([NSString stringWithFormat:@"fr: (%d, %d, %d, %d)", int(x + buffer), int(y + buffer), tile_size - 2 * buffer, tile_size - 2 * buffer], 3);
            new_squares.push_back(cv::Rect(int(x + buffer), int(y + buffer), tile_size - 2 * buffer, tile_size - 2 * buffer));
        }
        if (rows == n_rows) {
            break;
        }
    }
}


static string find_letters_in_image(cv::Mat& src_color, vector<cv::Rect>& all_squares, int rows, int cols) {
    @try {
        find_all_squares(src_color, all_squares, rows, cols);
        // Hmmm...we didn't find any so fall back to best guess.
        if (all_squares.size() == 0) {
            find_squares_ab(src_color, all_squares, rows, cols);
        }
        
        vector<cv::Mat> images;
        for(int i = 0; i < all_squares.size(); i++) {
            auto square_rect = all_squares[i];
            cv::Mat image = src_color(square_rect);
            images.push_back(image);
        }
        
        string final_match = "";
        for (int i = 0; i < images.size();i++) {
            cv::Mat color_tile = images[i];
            string letter = "oops";
            
            tuple<int,int,double> best = find_best_match(color_tile, alphabet_baseline_contours);
            int bli_area = get<0>(best);
            auto possibleBlank = false;
            if (bli_area > 0) {
                int idx = get<1>(best);
                double diff = get<2>(best);
                if (idx >= 0) {
                    // Compare with hand-tuned, arbitrary percetage.
                    if (diff <= 0.04) {
                        auto candidate = get<0>(alphabet_baseline_contours[idx]);
                        if (diff == 0)
                            DebugOutput([NSString stringWithFormat:@"%d: exact match", i]);
                        else
                            DebugOutput([NSString stringWithFormat:@"%d: close enough match: %.3f", i, diff]);
                        letter = candidate;
                    } else {
                        DebugOutput([NSString stringWithFormat:@"%d: match but not close enough: %.3f", i, diff]);
                        letter = "_";
                        possibleBlank = true;
                    }
                } else {
                    DebugOutput([NSString stringWithFormat:@"%d: shape but no match", i]);
                    letter = "_"; // ?";
                    possibleBlank = true;
                }
            } else {
                DebugOutput([NSString stringWithFormat:@"%d: no shape found", i]);
                letter = "_"; // ?";
                possibleBlank = true;
            }
            
            if (possibleBlank && is_blank_tile(color_tile)) {
                letter = "*";
            }
            final_match += letter;
        }
        return final_match;
    } @catch (NSException *exception) {
        return [[exception description] UTF8String];
    }
}

#pragma mark -

@implementation OpenCVWrapper

+ (NSString*) findLettersInMat: (cv::Mat&) bgrMat withRows: (int) rows withCols: (int) cols
{
    @try {
        create_ab_font_baseline();
        
        vector<cv::Rect> allSquares;
        string match = find_letters_in_image(bgrMat, allSquares, rows, cols);
        
        auto match2 = [NSString stringWithUTF8String:match.c_str()];
        for (int i = 0; i < allSquares.size(); i++) {
            auto rect = allSquares[i];
            match2 = [match2 stringByAppendingString:[NSString stringWithFormat:@";%d,%d,%d,%d", rect.tl().x, rect.tl().y, rect.width, rect.height]];
        }
        
        return match2;
    } @catch (NSException *exception) {
        return [exception description];
    } @finally {
        printParams();
    }
}

+ (NSString*) findLettersWithUrl: (NSURL*) inputUrl withRows: (int) rows withCols: (int) cols {
    @try {
        cv::Mat bgrMat = cv::imread(string([[inputUrl path] UTF8String]));
        if (bgrMat.size[0]==0 || bgrMat.size[1]==0) {
            return @"";
        }
        return [OpenCVWrapper findLettersInMat: bgrMat
                                      withRows: rows
                                      withCols: cols];
    } @catch (NSException *exception) {
        return [exception description];
    }
}



void _UIImageToMat(const UIImage* image, cv::Mat& m, bool alphaExist) {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGColorSpaceModel colorSpaceModel = CGColorSpaceGetModel(colorSpace);
    CGFloat cols = CGImageGetWidth(image.CGImage), rows = CGImageGetHeight(image.CGImage);
    CGContextRef contextRef;
    CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast;
    if (colorSpaceModel == kCGColorSpaceModelMonochrome)
    {
        m.create(rows, cols, CV_8UC1); // 8 bits per component, 1 channel
        bitmapInfo = kCGImageAlphaNone;
        if (!alphaExist)
            bitmapInfo = kCGImageAlphaNone;
        else
            m = cv::Scalar(0);
        
        contextRef = CGBitmapContextCreate(m.data, m.cols, m.rows, 8,
                                           m.step[0], colorSpace,
                                           bitmapInfo);
    }
    else
    {
        m.create(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
        if (!alphaExist)
            bitmapInfo = kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault;
        else
            m = cv::Scalar(0);
        contextRef = CGBitmapContextCreate(m.data, m.cols, m.rows, 8,
                                           m.step[0], colorSpace,
                                           bitmapInfo);
    }
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
}


+ (NSString*) findLettersInImage: (UIImage*) inputImage withRows: (int) rows withCols: (int) cols {
    @try {
        cv::Mat rgbaMat;
        _UIImageToMat(inputImage, rgbaMat, true);
        cv::Mat bgrMat;
        cv::cvtColor(rgbaMat, bgrMat, cv::COLOR_RGBA2BGR);
        return [OpenCVWrapper findLettersInMat: bgrMat
                                      withRows: rows
                                      withCols: cols];
    } @catch (NSException *exception) {
        return [exception description];
    }
}
@end

