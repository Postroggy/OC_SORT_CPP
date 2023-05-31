#include <iostream>
#include <vector>

void swap_adjacent_pairs(std::vector<int> &v) {
    if (v.size() < 2) return;
    for (auto i = begin(v), j = i + 1, e = end(v); j < e; i += 2, j += 2) {
        std::swap(*i, *j);
    }
}

int main() {
    std::vector<int> v{1, 2, 3, 4, 5, 6};
    swap_adjacent_pairs(v);
    for (int x: v) { std::cout << x << ' '; }
    std::cout << '\n';
}
