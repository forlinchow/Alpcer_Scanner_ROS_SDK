//
// Created by Zachary on 2024/6/18.
//

#ifndef LIVOXSCANNER_PLY_FAST_WRITER_H
#define LIVOXSCANNER_PLY_FAST_WRITER_H
#include <vector>
namespace pfw {
    typedef struct {//需要对齐
        float x;
        float y;
        float z;
        uint8_t intensity;
        uint8_t red;
        uint8_t green;
        uint8_t blue;
    } Point;

    class PlyFastWriter {
    private:
        std::vector<Point> data;
    public:
        inline void clear() {
            data.clear();
        }
        inline void reserve(size_t capacity) {
            data.reserve(capacity);
        }
        inline void addPoint(Point&& point) {
            data.emplace_back(point);
            //memcpy(reinterpret_cast<void *>(data.back()), (const void *) &point, 1);
        }
        void write(const std::string &filepath) {
            // Open stream for writing
            std::ofstream outStream(filepath, std::ios::out | std::ios::binary);
            if (!outStream.good()) {
                throw std::runtime_error("Ply fast writer: Could not open output file " + filepath + " for writing.");
            }
            writeHeader(outStream);
            writeData(outStream);
        }
    private:
        void writeHeader(std::ostream& outStream) {
            outStream << "ply\n";
            outStream << "format binary_little_endian 1.0\n";
            outStream << "comment Written with pct\n";
            outStream << "element vertex " << std::to_string(data.size()) << "\n";
            outStream << "property float x\n";
            outStream << "property float y\n";
            outStream << "property float z\n";
            outStream << "property uchar intensity\n";
            outStream << "property uchar red\n";
            outStream << "property uchar green\n";
            outStream << "property uchar blue\n";
            outStream << "end_header\n";
        }

        void writeData(std::ostream& outStream) {
            outStream.write((char*)data.data(), data.size()*sizeof(Point));
        }
    };
}

#endif //LIVOXSCANNER_PLY_FAST_WRITER_H
