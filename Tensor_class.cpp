#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

using namespace std;

std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> result;

    size_t pos = 0;
    while (pos < str.length()) {
        size_t start = str.find(delimiter, pos);
        if (start != std::string::npos) {
            result.push_back(str.substr(pos, start - pos));
            pos = start + delimiter.length();
        }
        else {
            result.push_back(str.substr(pos));
            break;
        }
    }

    return result;
}

std::vector<std::string> split_more(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> result;
    size_t pos = 0;
    while (pos != std::string::npos) {
        size_t next_pos = str.find(delimiter, pos);
        if (next_pos == std::string::npos) {
            result.push_back(str.substr(pos));
            break;
        }
        result.push_back(str.substr(pos, next_pos - pos));
        pos = next_pos + delimiter.size();
    }
    return result;
}

vector<vector<vector<double>>> parse_vector(const string& instr) {
    string str = instr.substr(1, instr.size() - 2);

    vector<vector<vector<double>>> result{};
    vector<string> res1 = {};
    vector<vector<string>> res2 = {};

    for (int i = 1; i < split(str, "{{").size(); i++) {
        string splitted = split(str, "{{")[i];
        if (i == split(str, "{{").size() - 1) {
            res1.push_back(splitted.substr(0, splitted.size() - 2));
        }
        else {
            res1.push_back(splitted.substr(0, splitted.size() - 4));
        }
    }

    for (int i = 0; i < res1.size(); i++) {
        res2.push_back(split_more(res1[i], "}, {"));
    }
    vector<vector<vector<double>>> res3(res1.size(), vector<vector<double>>(res2[0].size(), vector<double>({ 0.0 })));

    for (int i = 0; i < res2.size(); i++) {
        for (int j = 0; j < res2[0].size(); j++) {
            vector<string>splitted = split(res2[i][j], ", ");
            vector<double>final_res = {};
            for (int t = 0; t < splitted.size(); t++) {
                final_res.push_back(stod(splitted[t]));
            }
            res3[i][j] = final_res;

        }
    }

    return res3;
}

class Tensor
{
private:
    int x_size, y_size, z_size;
    vector<vector<vector<double>>> tensor;
public:

    Tensor(vector<vector<vector<double>>>);

    const vector<int> getSize() const {
        return vector<int> {z_size, y_size, x_size};
    }

    const vector<vector<vector<double>>>& getTensor() const {
        return tensor;
    }

    void setTensor(vector<vector<vector<double>>> new_tensor) {
        this->tensor = new_tensor;
        z_size = new_tensor.size();
        y_size = new_tensor[0].size();
        x_size = new_tensor[0][0].size();

    }

    Tensor setTensorValue(int z, int y, int x, double value) {
        Tensor result(tensor);
        result.tensor[z][y][x] = value;
        return result;
    }

    Tensor multiply(const Tensor& ten2) {
        if (this->x_size != ten2.y_size) {
            throw invalid_argument("Invalid sizes for matrix multiplication.");
        }

        vector<vector<vector<double>>> result(this->z_size, vector<vector<double>>(ten2.x_size, vector<double>(this->y_size, 0.0)));


        for (int z = 0; z < this->z_size; ++z) {
            for (int y = 0; y < this->y_size; ++y) {
                for (int x = 0; x < ten2.x_size; ++x) {
                    for (int k = 0; k < this->x_size; ++k) {
                        result[z][y][x] += this->tensor[z][y][k] * ten2.tensor[z][k][x];
                    }
                }
            }
        }

        return Tensor(result);
    }

    vector<vector<vector<vector<vector<vector<double>>>>>> multiply_6(const Tensor& ten2) {
        vector<vector<vector<vector<vector<vector<double>>>>>> result(this->z_size, vector<vector<vector<vector<vector<double>>>>>(this->y_size, vector<vector<vector<vector<double>>>>(this->x_size, vector<vector<vector<double>>>(ten2.z_size, vector<vector<double>>(ten2.y_size, vector<double>(ten2.x_size, 0.0))))));
        for (int n1 = 0; n1 < this->z_size; n1++) {
            for (int n2 = 0; n2 < this->y_size; n2++) {
                for (int n3 = 0; n3 < this->x_size; n3++) {
                    for (int m1 = 0; m1 < ten2.z_size; m1++) {
                        for (int m2 = 0; m2 < ten2.y_size; m2++) {
                            for (int m3 = 0; m3 < ten2.x_size; m3++) {
                                result[n1][n2][n3][m1][m2][m3] = this->tensor[n1][n2][n3] * ten2.tensor[m1][m2][m3];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    Tensor transpose() {
        vector<vector<vector<double>>> result(this->x_size, vector<vector<double>>(this->y_size, vector<double>(this->z_size, 0.0)));
        for (int z = 0; z < this->z_size; z++) {
            for (int y = 0; y < this->y_size; y++) {
                for (int x = 0; x < this->x_size; x++) {
                    result[x][y][z] = this->tensor[z][y][x];
                }
            }
        }
        return Tensor(result);
    }

    Tensor convolution() {
        vector<double> result(this->y_size, 0.0);
        for (int y = 0; y < this->y_size; y++) {
            for (int z = 0; z < this->z_size; z++) {
                for (int x = 0; x < this->x_size; x++) {
                    result[y] += this->tensor[z][y][x];
                }
            }
        }
        return Tensor({ {result} });
    }


    Tensor operator*(const Tensor& ten2) {
        if (this->x_size != ten2.y_size) {
            throw invalid_argument("Invalid sizes for matrix multiplication.");
        }

        vector<vector<vector<double>>> result(this->z_size, vector<vector<double>>(this->y_size, vector<double>(this->x_size, 0.0)));


        for (int z = 0; z < this->z_size; ++z) {
            for (int y = 0; y < this->y_size; ++y) {
                for (int x = 0; x < this->x_size; ++x) {
                    result[z][y][x] = this->tensor[z][y][x] * ten2.tensor[z][y][x];
                }
            }
        }

        return Tensor(result);
    }

    Tensor operator*(const double& scalar) {
        vector<vector<vector<double>>> result(this->z_size, vector<vector<double>>(this->y_size, vector<double>(this->x_size, 0.0)));
        for (int z = 0; z < this->z_size; z++) {
            for (int y = 0; y < this->y_size; y++) {
                for (int x = 0; x < this->x_size; x++) {
                    result[z][y][x] = this->tensor[z][y][x] * scalar;
                }
            }
        }
        return Tensor(result);
    }

    Tensor operator+(const double& scalar) {
        vector<vector<vector<double>>> result(this->z_size, vector<vector<double>>(this->y_size, vector<double>(this->x_size, 0.0)));
        for (int z = 0; z < this->z_size; z++) {
            for (int y = 0; y < this->y_size; y++) {
                for (int x = 0; x < this->x_size; x++) {
                    result[z][y][x] = this->tensor[z][y][x] + scalar;
                }
            }
        }
        return Tensor(result);
    }

    Tensor operator+(const Tensor& scalar) {
        vector<vector<vector<double>>> result(this->z_size, vector<vector<double>>(this->y_size, vector<double>(this->x_size, 0.0)));
        for (int z = 0; z < this->z_size; z++) {
            for (int y = 0; y < this->y_size; y++) {
                for (int x = 0; x < this->x_size; x++) {
                    result[z][y][x] = this->tensor[z][y][x] + scalar.tensor[z][y][x];
                }
            }
        }
        return Tensor(result);
    }

    friend istream& operator>>(istream& in, Tensor& t) {
        string line;
        getline(in, line);
        vector<vector<vector<double>>> parsed = parse_vector(line);
        t = Tensor(vector<vector<vector<double>>>(parsed.size(), vector<vector<double>>(parsed[0].size(), vector<double>(parsed[0][0].size(), 0.0))));
        t.setTensor(parsed);

        return in;
    }

    Tensor symmetrize_3() {
        vector<vector<vector<double>>> out_tensor = this->tensor;
        for (int i = 0; i < z_size; i++) {
            for (int j = i; j < y_size; j++) {
                for (int k = j; k < x_size; k++) {
                    double temp = (out_tensor[i][j][k] + out_tensor[i][k][j] + out_tensor[j][i][k] + out_tensor[j][k][i] + out_tensor[k][i][j] + out_tensor[k][j][i]) / 6.0;
                    out_tensor[i][j][k] = out_tensor[i][k][j] = out_tensor[j][i][k] = out_tensor[j][k][i] = out_tensor[k][i][j] = out_tensor[k][j][i] = temp;
                }
            }
        }
        return Tensor(out_tensor);
    }

    Tensor symmetrize_2(int dim1, int dim2) {
        vector<vector<vector<double>>> tensor = this->tensor;


        for (int k = 0; k < z_size; k++) {
            for (int i = 0; i < x_size; i++) {
                for (int j = i; j < y_size; j++) {
                    double temp = (tensor[k][i][j] + tensor[k][j][i]) / 2.0;
                    tensor[k][i][j] = tensor[k][j][i] = temp;
                }
            }
        }

        for (int k = 0; k < z_size; k++) {
            for (int i = 0; i < y_size; i++) {
                for (int j = i; j < x_size; j++) {
                    double temp = (tensor[k][i][j] + tensor[k][j][i]) / 2.0;
                    tensor[k][i][j] = tensor[k][j][i] = temp;
                }
            }
        }

        return Tensor(tensor);
    }

    Tensor anty_symmetrize_3() {
        vector<vector<vector<double>>> result = this->tensor;
        for (int k = 0; k < z_size; k++) {
            for (int l = 0; l < y_size; l++) {
                for (int m = 0; m < x_size; m++) {
                    result[k][l][m] = (tensor[k][l][m] + tensor[l][m][k] + tensor[m][k][l] - tensor[k][m][l] - tensor[l][k][m] - tensor[m][l][k]) / 6;
                }
            }
        }
        return Tensor(result);
    }

};

ostream& operator<<(ostream& os, const Tensor& ten) {
    os << "{";
    for (int z = 0; z < ten.getSize()[0]; z++) {
        os << "\n {";
        for (int y = 0; y < ten.getSize()[1]; y++) {
            os << "\n  {";
            for (int x = 0; x < ten.getSize()[2]; x++) {
                os << " " << ten.getTensor()[z][y][x];
            }
            os << " }";
        }
        os << "\n }";
    }
    os << "\n}";
    return os;
}

ostream& operator<<(ostream& os, const vector<vector<vector<vector<vector<vector<double>>>>>>& ten) {
    os << "{" << endl;
    for (int i = 0; i < ten.size(); i++) {
        os << "  {" << endl;
        for (int j = 0; j < ten[i].size(); j++) {
            os << "    {" << endl;
            for (int k = 0; k < ten[i][j].size(); k++) {
                os << "      {" << endl;
                for (int l = 0; l < ten[i][j][k].size(); l++) {
                    os << "        {" << endl;
                    for (int m = 0; m < ten[i][j][k][l].size(); m++) {
                        os << "          { ";
                        for (int n = 0; n < ten[i][j][k][l][m].size(); n++) {
                            os << ten[i][j][k][l][m][n] << " ";
                        }
                        os << "}," << endl;
                    }
                    os << "        }," << endl;
                }
                os << "      }," << endl;
            }
            os << "    }," << endl;
        }
        os << "  }," << endl;
    }
    os << "}" << endl;
    return os;
}


Tensor::Tensor(vector<vector<vector<double>>> arr) {
    tensor = arr;
    z_size = tensor.size();
    y_size = tensor[0].size();
    x_size = tensor[0][0].size();
}


int main() {
    Tensor t1({ {{2, 1}, {4, 1}}, {{1, 1}, {10, 1}} });
    cout << t1.multiply_6(t1);
    return 0;
}
