# Docker를 이용한 Hadoop 분산 Cluster - Spark 환경과 Kafka 분산 broker환경 구축 및 연동

Hadoop은 빅데이터 처리를 HDFS라는 디스크 위에서 맵리듀스를 통해 데이터셋을 쪼개 병렬처리하게 된다
하지만 HDFS를 사용하기 위해서는 Disk I/O는 불가피한데 Disk I/O는 다들 알다시피 보조 저장장치에서 발생하기 때문에 속도의 한계가 있다
이 때 인 메모리 방식으로 더욱 빠른 속도로 데이터를 분산 처리하는 spark가 등장하여 맵리듀스를 대체하게 된다.
또한 스파크는 스파크 스트리밍이라는 실시간 데이터 처리에도 사용될 수 있어 실시간 대쉬보드에 쓰일 데이터를 처리하면서 HDFS에도 데이터를 저장할 수 있는 환경을 마련했다

카프카는 실시간으로 기록 스트림을 게시, 구독, 저장 및 처리할 수 있는 분산 데이터 스트리밍 플랫폼이다
N대의 데이터 공급자와 M대의 데이터 소비자를 연결시켜 필요한 곳에 실시간으로 공급하는 역할을 한다

현재 하나의 AWS 서버를 갖고 있고 Hadoop과 Kafka를 분산 환경에서 구동시키기 위해 docker를 사용하여 해당 환경을 구축하기로 했다

이 과정은 https://1mini2.tistory.com/99 를 참조하여 만들어졌다

먼저 도커를 사용해 master 서버와 worker 1~3 서버를 만들어 준다
서버 환경을 직접 구축해 볼 것이므로 다른 커스텀 이미지를 사용하지 않고 우분투 공식 이미지를 사용한다

```
-- terminal1
docker run -it --name master --hostname master ubuntu
-- terminal2
docker run -it --name worker1 --hostname worker1 ubuntu
-- terminal3
docker run -it --name worker2 --hostname worker2 ubuntu
-- terminal4
docker run -it --name worker3 --hostname worker3 ubuntu
```

위의 명령어를 실행시키면 ubuntu 이미지를 다운로드하고 이 이미지로 master,worker 1~3 컨테이너를 만들고 bash쉘에 접속한다
같은 이미지는 한 번만 다운로드된다

**
유의할 점
**

1. 분산 시스템을 구축하기 위해서는 각각의 서버가 ssh를 통해 통신하여야한다
   kafka 분산시스템을 만들 때 ssh를 간과하여 한참 헤맸다
2. hostname을 가급적 먼저 수정하여 설정파일에 들어갈 hostname을 나중에 수정하지 않게끔 하자
3. 스칼라는 아직까진 2.12버전이 많이 쓰이는 것 같다(maven plugin도 2.12버전을 지원하는 것이 많다)
4. hadoop 분산 시스템을 초기화 할 때는 datanode 디렉토리를 완전히 삭제한 후 `hdfs namenode -format`을 실행시켜야 datanode와 namenode가 동일한 clusterID를 갖게된다
