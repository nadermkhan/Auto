// smart_mouse.cpp - Compile: g++ smart_mouse.cpp -o smart_mouse -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lX11 -lXtst -ltesseract -std=c++17
// Windows: Use Windows.h instead of X11, link against appropriate libs

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>
#include <thread>
#include <chrono>

#ifdef _WIN32
    #include <windows.h>
    #pragma comment(lib, "user32.lib")
#else
    #include <X11/Xlib.h>
    #include <X11/Xutil.h>
    #include <X11/extensions/XTest.h>
#endif

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

// ============================================================================
// CROSS-PLATFORM SCREEN CAPTURE & MOUSE CONTROL
// ============================================================================

class ScreenController {
private:
#ifdef _WIN32
    HDC hScreen;
    HDC hDC;
    HBITMAP hBitmap;
    int screenWidth, screenHeight;
#else
    Display* display;
    Window root;
    int screenWidth, screenHeight;
#endif

public:
    ScreenController() {
#ifdef _WIN32
        hScreen = GetDC(NULL);
        screenWidth = GetSystemMetrics(SM_CXSCREEN);
        screenHeight = GetSystemMetrics(SM_CYSCREEN);
#else
        display = XOpenDisplay(nullptr);
        if (!display) throw std::runtime_error("Cannot open display");
        root = DefaultRootWindow(display);
        Screen* screen = DefaultScreenOfDisplay(display);
        screenWidth = screen->width;
        screenHeight = screen->height;
#endif
    }

    ~ScreenController() {
#ifdef _WIN32
        ReleaseDC(NULL, hScreen);
#else
        if (display) XCloseDisplay(display);
#endif
    }

    cv::Mat captureScreen() {
#ifdef _WIN32
        hDC = CreateCompatibleDC(hScreen);
        hBitmap = CreateCompatibleBitmap(hScreen, screenWidth, screenHeight);
        SelectObject(hDC, hBitmap);
        BitBlt(hDC, 0, 0, screenWidth, screenHeight, hScreen, 0, 0, SRCCOPY);

        BITMAPINFOHEADER bi = {sizeof(BITMAPINFOHEADER), screenWidth, -screenHeight, 1, 32, BI_RGB};
        cv::Mat mat(screenHeight, screenWidth, CV_8UC4);
        GetDIBits(hDC, hBitmap, 0, screenHeight, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
        
        DeleteObject(hBitmap);
        DeleteDC(hDC);
        
        cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
        return mat;
#else
        XImage* img = XGetImage(display, root, 0, 0, screenWidth, screenHeight, AllPlanes, ZPixmap);
        cv::Mat mat(screenHeight, screenWidth, CV_8UC4, img->data);
        cv::Mat result;
        cv::cvtColor(mat, result, cv::COLOR_BGRA2BGR);
        XDestroyImage(img);
        return result.clone();
#endif
    }

    void moveMouse(int x, int y) {
#ifdef _WIN32
        SetCursorPos(x, y);
#else
        XWarpPointer(display, None, root, 0, 0, 0, 0, x, y);
        XFlush(display);
#endif
    }

    void click(int x, int y, bool rightClick = false) {
        moveMouse(x, y);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
#ifdef _WIN32
        DWORD downFlag = rightClick ? MOUSEEVENTF_RIGHTDOWN : MOUSEEVENTF_LEFTDOWN;
        DWORD upFlag = rightClick ? MOUSEEVENTF_RIGHTUP : MOUSEEVENTF_LEFTUP;
        mouse_event(downFlag, 0, 0, 0, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        mouse_event(upFlag, 0, 0, 0, 0);
#else
        unsigned int button = rightClick ? Button3 : Button1;
        XTestFakeButtonEvent(display, button, True, CurrentTime);
        XFlush(display);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        XTestFakeButtonEvent(display, button, False, CurrentTime);
        XFlush(display);
#endif
    }

    void doubleClick(int x, int y) {
        click(x, y);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        click(x, y);
    }

    std::pair<int, int> getScreenSize() {
        return {screenWidth, screenHeight};
    }
};

// ============================================================================
// UI ELEMENT DETECTION
// ============================================================================

struct UIElement {
    cv::Rect bounds;
    std::string text;
    std::string type; // "button", "text", "icon", "input"
    float confidence;
    cv::Point center() const { return cv::Point(bounds.x + bounds.width/2, bounds.y + bounds.height/2); }
};

class SmartVision {
private:
    tesseract::TessBaseAPI* ocr;
    
    // Detect button-like regions using edge detection and contours
    std::vector<cv::Rect> detectButtonRegions(const cv::Mat& img) {
        cv::Mat gray, edges, dilated;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        
        // Edge detection
        cv::Canny(gray, edges, 50, 150);
        cv::dilate(edges, dilated, cv::Mat(), cv::Point(-1,-1), 2);
        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::vector<cv::Rect> buttons;
        for (const auto& contour : contours) {
            cv::Rect rect = cv::boundingRect(contour);
            // Filter by size and aspect ratio (typical button characteristics)
            if (rect.width > 40 && rect.width < 400 && 
                rect.height > 20 && rect.height < 100 &&
                rect.width > rect.height) {
                buttons.push_back(rect);
            }
        }
        return buttons;
    }

    // Detect text regions and extract text
    std::vector<UIElement> detectTextRegions(const cv::Mat& img) {
        std::vector<UIElement> elements;
        
        // OCR on full image
        ocr->SetImage(img.data, img.cols, img.rows, 3, img.step);
        ocr->Recognize(0);
        
        tesseract::ResultIterator* ri = ocr->GetIterator();
        tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
        
        if (ri != 0) {
            do {
                const char* word = ri->GetUTF8Text(level);
                if (word == nullptr) continue;
                
                float conf = ri->Confidence(level);
                int x1, y1, x2, y2;
                ri->BoundingBox(level, &x1, &y1, &x2, &y2);
                
                UIElement elem;
                elem.bounds = cv::Rect(x1, y1, x2-x1, y2-y1);
                elem.text = word;
                elem.confidence = conf;
                elem.type = "text";
                
                elements.push_back(elem);
                delete[] word;
            } while (ri->Next(level));
            delete ri;
        }
        
        return elements;
    }

    // Color-based region detection (for buttons/UI elements)
    std::vector<cv::Rect> detectColorRegions(const cv::Mat& img, cv::Scalar targetColor, int tolerance = 30) {
        cv::Mat hsv, mask;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        
        cv::Scalar lower(targetColor[0] - tolerance, 100, 100);
        cv::Scalar upper(targetColor[0] + tolerance, 255, 255);
        cv::inRange(hsv, lower, upper, mask);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::vector<cv::Rect> regions;
        for (const auto& contour : contours) {
            cv::Rect rect = cv::boundingRect(contour);
            if (rect.width > 20 && rect.height > 20) {
                regions.push_back(rect);
            }
        }
        return regions;
    }

public:
    SmartVision() {
        ocr = new tesseract::TessBaseAPI();
        if (ocr->Init(NULL, "eng")) {
            throw std::runtime_error("Could not initialize tesseract");
        }
    }

    ~SmartVision() {
        ocr->End();
        delete ocr;
    }

    std::vector<UIElement> analyzeScreen(const cv::Mat& screenshot) {
        std::vector<UIElement> allElements;
        
        // Detect text elements
        auto textElements = detectTextRegions(screenshot);
        allElements.insert(allElements.end(), textElements.begin(), textElements.end());
        
        // Detect button-like regions
        auto buttonRects = detectButtonRegions(screenshot);
        for (const auto& rect : buttonRects) {
            UIElement elem;
            elem.bounds = rect;
            elem.type = "button";
            elem.confidence = 0.7f;
            
            // Try to extract text from button region
            cv::Mat roi = screenshot(rect);
            ocr->SetImage(roi.data, roi.cols, roi.rows, 3, roi.step);
            char* text = ocr->GetUTF8Text();
            if (text) {
                elem.text = text;
                delete[] text;
            }
            
            allElements.push_back(elem);
        }
        
        return allElements;
    }

    // Fuzzy text matching
    float textSimilarity(const std::string& a, const std::string& b) {
        std::string lowerA = a, lowerB = b;
        std::transform(lowerA.begin(), lowerA.end(), lowerA.begin(), ::tolower);
        std::transform(lowerB.begin(), lowerB.end(), lowerB.begin(), ::tolower);
        
        if (lowerA.find(lowerB) != std::string::npos || lowerB.find(lowerA) != std::string::npos) {
            return 0.9f;
        }
        
        // Simple Levenshtein-like scoring
        int matches = 0;
        for (char c : lowerA) {
            if (lowerB.find(c) != std::string::npos) matches++;
        }
        return (float)matches / std::max(lowerA.length(), lowerB.length());
    }

    UIElement* findBestMatch(std::vector<UIElement>& elements, const std::string& query) {
        UIElement* best = nullptr;
        float bestScore = 0.0f;
        
        for (auto& elem : elements) {
            float score = textSimilarity(elem.text, query);
            
            // Boost score for buttons when looking for clickable elements
            if (elem.type == "button") score *= 1.2f;
            
            // Boost score based on OCR confidence
            score *= (elem.confidence / 100.0f);
            
            if (score > bestScore) {
                bestScore = score;
                best = &elem;
            }
        }
        
        return (bestScore > 0.3f) ? best : nullptr;
    }
};

// ============================================================================
// SMART MOUSE AUTOMATION ENGINE
// ============================================================================

class SmartMouse {
private:
    ScreenController screen;
    SmartVision vision;
    cv::Mat lastScreenshot;
    std::vector<UIElement> lastElements;

public:
    void updateScreen() {
        lastScreenshot = screen.captureScreen();
        lastElements = vision.analyzeScreen(lastScreenshot);
        std::cout << "Detected " << lastElements.size() << " UI elements\n";
    }

    void showDetections() {
        cv::Mat display = lastScreenshot.clone();
        for (const auto& elem : lastElements) {
            cv::rectangle(display, elem.bounds, cv::Scalar(0, 255, 0), 2);
            cv::putText(display, elem.text + " (" + elem.type + ")", 
                       cv::Point(elem.bounds.x, elem.bounds.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        cv::imshow("Detected Elements", display);
        cv::waitKey(0);
    }

    bool clickOn(const std::string& target, bool rightClick = false) {
        updateScreen();
        
        UIElement* elem = vision.findBestMatch(lastElements, target);
        if (elem) {
            std::cout << "Clicking on: " << elem->text << " at (" 
                     << elem->center().x << ", " << elem->center().y << ")\n";
            screen.click(elem->center().x, elem->center().y, rightClick);
            return true;
        }
        
        std::cout << "Could not find element matching: " << target << "\n";
        return false;
    }

    bool doubleClickOn(const std::string& target) {
        updateScreen();
        
        UIElement* elem = vision.findBestMatch(lastElements, target);
        if (elem) {
            std::cout << "Double-clicking on: " << elem->text << "\n";
            screen.doubleClick(elem->center().x, elem->center().y);
            return true;
        }
        return false;
    }

    void moveTo(const std::string& target) {
        updateScreen();
        
        UIElement* elem = vision.findBestMatch(lastElements, target);
        if (elem) {
            std::cout << "Moving to: " << elem->text << "\n";
            screen.moveMouse(elem->center().x, elem->center().y);
        }
    }

    // Interactive command mode
    void commandMode() {
        std::string cmd, target;
        
        std::cout << "\n=== Smart Mouse Control ===\n";
        std::cout << "Commands:\n";
        std::cout << "  click <text>       - Click on element containing text\n";
        std::cout << "  right <text>       - Right-click on element\n";
        std::cout << "  double <text>      - Double-click on element\n";
        std::cout << "  move <text>        - Move mouse to element\n";
        std::cout << "  show               - Show detected elements\n";
        std::cout << "  refresh            - Refresh screen analysis\n";
        std::cout << "  quit               - Exit\n\n";
        
        while (true) {
            std::cout << "> ";
            std::cin >> cmd;
            
            if (cmd == "quit") break;
            else if (cmd == "show") {
                updateScreen();
                showDetections();
            }
            else if (cmd == "refresh") {
                updateScreen();
            }
            else if (cmd == "click") {
                std::getline(std::cin >> std::ws, target);
                clickOn(target);
            }
            else if (cmd == "right") {
                std::getline(std::cin >> std::ws, target);
                clickOn(target, true);
            }
            else if (cmd == "double") {
                std::getline(std::cin >> std::ws, target);
                doubleClickOn(target);
            }
            else if (cmd == "move") {
                std::getline(std::cin >> std::ws, target);
                moveTo(target);
            }
            else {
                std::cout << "Unknown command\n";
            }
        }
    }
};

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    try {
        SmartMouse mouse;
        
        if (argc > 1) {
            // Command-line mode
            std::string action = argv[1];
            if (action == "click" && argc > 2) {
                mouse.clickOn(argv[2]);
            } else if (action == "show") {
                mouse.updateScreen();
                mouse.showDetections();
            }
        } else {
            // Interactive mode
            mouse.commandMode();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
