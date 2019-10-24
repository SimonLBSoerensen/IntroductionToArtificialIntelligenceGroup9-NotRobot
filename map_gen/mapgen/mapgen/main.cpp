#include <iostream>
#include <thread>
#include <vector>
#include "util.h"
#include "sokoban.h"
#include "cxxopts.h"

using namespace std;

void printLevel(const Table<char>& level) {
  for (int y : level.getBounds().getYRange()) {
    for (int x : level.getBounds().getXRange())
      cout << level[Vec2(x, y)];
    cout << '\n';
  }
}

vector<vector<char>> levelToVec(const Table<char>& level) {
	vector<vector<char>> map;
	for (int y : level.getBounds().getYRange()) {
		map.push_back(vector<char>());
		for (int x : level.getBounds().getXRange())
			map[map.size() - 1].push_back(level[Vec2(x, y)]);
	}
	return map;
}

void printLevel(const vector<vector<char>>& level) {
	for (auto row : level) {
		for (auto c : row) cout << c;
		cout << endl;
	}
}

vector<vector<char>> genMap(int width, int height, int numTries,
	int numBoulders, int numMoves, int rooms, int doors) {
	
	RandomGen randomGen;
	randomGen.init(time(0));

	Vec2 levelSize(width, height);
	
	SokobanMaker sokoban(randomGen, levelSize, numBoulders, numMoves);
	sokoban.setNumRooms(rooms);
	sokoban.setNumDoors(doors);
	int maxDepth = -1;
	while (numTries--)
	{
		if (sokoban.make() && sokoban.getMaxDepth() > maxDepth) {
			Table<char> map = sokoban.getResult();
			vector<vector<char>> vec_map = levelToVec(map);
			return vec_map;
		}
	}
	cout << "Unable to generate a level with these parameters" << endl;
	return vector<vector<char>>();
}




void trySokoban(RandomGen& randomGen, Vec2 levelSize, int numTries,
                int numBoulders, int numMoves, int rooms, int doors) {
  int maxDepth = -1;
  for (int i = 0; i < numTries; ++i) {
    SokobanMaker sokoban(randomGen, levelSize, numBoulders, numMoves);
    sokoban.setNumRooms(rooms);
    sokoban.setNumDoors(doors);
    if (sokoban.make() && sokoban.getMaxDepth() > maxDepth) {
      maxDepth = sokoban.getMaxDepth();
      cout << "Depth reached: " << maxDepth << endl;
	  Table<char> map = sokoban.getResult();
      printLevel(map);

	  //vector<vector<char>> vec_map = levelToVec(map);

    }
  }
  if (maxDepth == -1)
    cout << "Unable to generate a level with these parameters" << endl;
}

int main(int argc, char* argv[]) {
  int tries = 1000;
  int width = 28;
  int height = 28;
  int boulders = 5;
  int moves = 100;
  int rooms = 100;
  int doors = 0;
  RandomGen randomGen;
  randomGen.init(time(0));

  Vec2 levelsize(width, width);

  trySokoban(randomGen, levelsize, tries, boulders, moves, rooms, doors);

  //vector<vector<char>> map = genMap(width, height, tries, boulders, moves, rooms, doors);
  //printLevel(map);
}
