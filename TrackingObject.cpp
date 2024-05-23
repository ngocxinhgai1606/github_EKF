float frequencyLaser = 30; //hz
    float dt = 1.0/frequencyLaser ; // khoảng thời gian giữa hai dữ liệu liên tiếp
    int measurementSize = 4643;    // data size
    int groundDataSize = measurementSize;
    int frequencyFrame = 30; //(fps)
    int timeLate = 300; //ms
    int predictNumberValues = 9; //(frequencyFrame*timeLate/1000);