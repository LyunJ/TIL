# Docker를 이용한 Hadoop 분산 Cluster - Spark 환경과 Kafka 분산 broker환경 구축 및 연동

Hadoop은 빅데이터 처리를 HDFS라는 디스크 위에서 맵리듀스를 통해 데이터셋을 쪼개 병렬처리하게 된다
하지만 HDFS를 사용하기 위해서는 Disk I/O는 불가피한데 Disk I/O는 다들 알다시피 보조 저장장치에서 발생하기 때문에 속도의 한계가 있다
이 때 인 메모리 방식으로 더욱 빠른 속도로 데이터를 분산 처리하는 spark가 등장하여 맵리듀스를 대체하게 된다.
또한 스파크는 스파크 스트리밍이라는 실시간 데이터 처리에도 사용될 수 있어 실시간 대쉬보드에 쓰일 데이터를 처리하면서 HDFS에도 데이터를 저장할 수 있는 환경을 마련했다

카프카는 실시간으로 기록 스트림을 게시, 구독, 저장 및 처리할 수 있는 분산 데이터 스트리밍 플랫폼이다
N대의 데이터 공급자와 M대의 데이터 소비자를 연결시켜 필요한 곳에 실시간으로 공급하는 역할을 한다

현재 하나의 AWS 서버를 갖고 있고 Hadoop과 Kafka를 분산 환경에서 구동시키기 위해 docker를 사용하여 해당 환경을 구축하기로 했다

이 과정은 https://1mini2.tistory.com/99 를 참조하여 만들어졌다

## Docker를 활용한 Hadoop 분산 시스템

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

### master,worker 공통 설정

#### Java 설치

```
sudo apt update
sudo apt-get install openjdk-8-jdk -y
```

java 설치 후 확인

```
java -version
```

#### Hadoop 설치(파일명은 정확하지 않을 수 있음)

하둡을 설치하기 전에 외부 프로그램들을 설치할 폴더를 만든다

```
mkdir /opt
```

하둡을 다운받아 압축을 풀어준다

```
cd /opt
wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz
tar -zxvf hadoop-3.3.1.tar.gz
```

버전 변경의 편의와 경로 설정의 편의와 일관성을 위해 링크 설정을 해준다

```
ln -s hadoop-3.3.1 hadoop  //링크설정
chmod -R 775 hadoop-3.3.1
chown -R root.root hadoop-3.3.1
```


#### Scala 설치(파일명은 정확하지 않을 수 있음)

scala 2.12를 다운받아 압축을 풀어준다
maven의 대부분의 플러그인은 아직 2.13보다 2.12를 지원하는 것이 많다.

```
wget https://github.com/scala/scala/archive/v2.12.14.tar.gz
tar -zxvf v2.12.14.tar.gz
```

버전 변경의 편의와 경로 설정의 편의와 일관성을 위해 링크 설정을 해준다

```
ln -s scala-2.12 scala  //링크설정 
chmod -R 775 scala-2.12
chown -R root.root scala-2.12
```

#### Spark 설치(파일명은 정확하지 않을 수 있음)

spark를 다운받아 압축을 풀어준다.

```
wget https://dlcdn.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
tar -zxvf spark-3.1.2-bin-hadoop3.2.tgz
```

버전 변경의 편의와 경로 설정의 편의와 일관성을 위해 링크 설정을 해준다

```
ln -s spark-3.1.2-bin-hadoop3.2 spark  //링크설정 
chmod -R 775 spark-3.1.2-bin-hadoop3.2
chown -R root.root spark-3.1.2-bin-hadoop3.2
```

### 환경변수 설정 및 설정파일 수정

#### 환경변수
모든 계정에 적용이 될 환경변수를 설정해준다

```
vim /etc/profile //모든 계정에 적용이 될 환경설정
// 아래 문자열 추가
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export HADOOP_HOME=/opt/hadoop
export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export SPARK_HOME=/opt/spark
```

#### 하둡 설정파일

하둡 디렉토리로 이동

```
cd /opt/hadoop
```

##### core-site.xml
```
vim etc/hadoop/core-site.xml
// <configuration>내부에 아래 문자열 추가
   <property>
        <name>fs.defaultFS</name>
       <value>hdfs://master:9000</value>
   </property>         i
```

##### hdfs-site.xml

```
vim etc/hadoop/hdfs-site.xml
// <configuration>내부에 아래 문자열 추가
   <property>
        <name>dfs.replication</name>
        <value>3</value> 
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:///hdfs_dir/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:///hdfs_dir/datanode</value>
    </property>
    <property>
        <name>dfs.namenode.secondary.http-address</name>
        <value>worker1:50090</value>
   </property>
```
이 때, dfs.replication의 value는 worker의 개수만큼 설정한다
dfs.namenode.secondary.http-address의 worker1는 hostname으로서 미리 /etc/hosts에 ip 정보를 등록해놓아야한다.

##### yarn-site.xml

```
vim etc/hadoop/yarn-site.xml
// <configuration>내부에 아래 문자열 추가
   <property>
        <name>yarn.nodemanager.local-dirs</name>
        <value>file:///hdfs_dir/yarn/local</value>
    </property>
    <property>
        <name>yarn.nodemanager.log-dirs</name>
        <value>file:///hdfs_dir/yarn/logs</value>
    </property>
    <property>
        <name>yarn.resourcemanager.hostname</name>
         <value>master</value>
   </property>
```

##### mapred-site.xml

```
vim etc/hadoop/mapred-site.xml
// <configuration>내부에 아래 문자열 추가
   <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
```

##### hadoop-env.sh

```
vim etc/hadoop/hadoop-env.sh
// 아래 문자열 추가
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export HDFS_NAMENODE_USER="root"
export HDFS_DATANODE_USER="root"
export HDFS_SECONDARYNAMENODE_USER="root"
export YARN_RESOURCEMANAGER_USER="root"
export YARN_NODEMANAGER_USER="root"
```

#### Spark 설정파일

##### spark-default.conf

spark 경로로 이동하자

```
cd /opt/spark
```

spark-defaults.conf 파일을 복사하고 맨 아래에 세줄을 추가한다

```
cp conf/spark-defaults.conf.template conf/spark-defaults.conf
vim conf/spark-defaults.conf
// 아래 문자열 추가
// 주석 처리된 두 줄은 오류 때문에 해놓은 것
spark.master yarn
# spark.eventLog.enabled true
# spark.eventLog.dir hdfs://namenode:8021/spark_enginelog
```

##### spakr-env.sh

spark-env.sh 파일을 복사하고, 아래 다섯줄을 추가한다

```
cp conf/spark-env.sh.template conf/spark-env.sh
vim conf/spark-env.sh
// 아래 문자열 추가
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export SPARK_MASTER_HOST=master
export HADOOP_HOME=/opt/hadoop
export YARN_CONF_DIR=\$HADOOP_HOME/etc/hadoop
export HADOOP_CONF_DIR=\$HADOOP_HOME/etc/hadoop
```

### Master 서버 설정

```
vim /etc/hosts
// 아래 문자열 추가
(master ip) master
(worker1 ip) worker1
(worker2 ip) worker2
(worker3 ip) worker3
```

### Worker 서버 설정

```
vim /etc/hosts
// 아래 문자열 추가
(master ip) master
(worker1 ip) worker1
(worker2 ip) worker2
(worker3 ip) worker3
```

### SSH-KEY 교환하기(모든 서버에 적용)

ssh 설치

```
apt-get install ssh
```

```
vim /etc/ssh/sshd_config
// 38번째 줄의 PermitRootLogin의 주석을 해제
PermitsRootLogin yes
// 65번째 줄의 PasswordAuthentication를 yes로 변경한다
PasswordAuthentication yes
```

```
/etc/init.d/sshd restart

// 각각의 서버의 비밀번호를 설정해주자
passwd

//ssh-key를 생성한다 (모두 엔터를 누르자)
ssh-keygen
```

#### ssh 키 교환하기

모든 서버에서 아래의 명령어를 실행시켜주자

```
ssh-copy-id root@master
ssh-copy-id root@worker1
ssh-copy-id root@worker2
ssh-copy-id root@worker3
```

### HDFS 포맷하기

#### NameNode 포맷

마스터 노드와 Worker1노트에서 아래의 명령어를 실행하여 네임노드 포맷을 시작한다

```
/opt/hadoop/bin/hdfs namenode -format /hdfs_dir
```

#### DataNode 포맷

데이터 노드(worker1,worker2,worker3) 세 대에서도 아래의 명령어를 실행한다

```
/opt/hadoop/bin/hdfs datanode -format /hdfs_dir/
```

### HDFS & YARN 시작하기

실행하기 전에 Master 노드에서 설정파일을 하나 수정한다

```
vim /opt/hadoop/etc/hadoop/workers
// 아래의 문자열 추가
worker1
worker2
worker3
```

마스터 노드에서만 명령어를 실행해주자

```
/opt/hadoop/sbin/start-all.sh
```

jps 명령어로 HDFS&YARN이 잘 실행되었는지 확인 가능

```
jps
```

### SPARK 실행하기

HDFS와 같이 master 노드의 설정파일을 하나 수정한다

```
vim /opt/spark/conf/workers
//아래의 문자열을 추가한다
worker1
worker2
worker3
```

실행

```
/opt/spark/sbin/start-all.sh
```

## KAFKA 클러스터 구축
broker:3대

```
// terminal1
docker run -it --name broker1 --hostname broker1 ubuntu
// terminal2
docker run -it --name broker2 --hostname broker2 ubuntu
// terminal3
docker run -it --name broker3 --hostname broker3 ubuntu
```

Hadoop을 설치했던 대로 그대로 진행해준다

```
cd /opt
wget https://mirror.navercorp.com/apache/kafka/2.8.0/kafka_2.12-2.8.0.tgz
tar -zxvf kafka_2.12-2.8.0.tgz
ln -s kafka_2.12-2.8.0 kafka
chmod -R 775 kafka_2.12-2.8.0
chown -R root.root kafka_2.12-2.8.0
```

### 주키퍼 설정

```
cd /opt/kafka
vim config/zookeeper.properties
// 아래의 문자열 추가
initLimit=5
syncLimit=2

server.1=broker1:2888:3888
server.2=broker2:2888:3888
server.3=broker3:2888:3888
```

주키퍼 데이터 디렉토리 생성

```
mkdir /tmp/zookeeper
```

각 서버마다 주키퍼 id 부여

```
// broker 번호별로 각기 다른 id 부여 (broker1의 경우)
echo 1 > /tmp/zookeeper/myid
```

### 카프카 설정

브로커 마다 각기 다른 id와 hostname 설정

```
vim config/server.properties
// 아래의 문자열 추가 (broker1의 경우)
broker.id=1
listeners=PLAINTEXT://:9092
advertised.listeners=PLAINTEXT://broker-server1:9092
zookeeper.connect=broker1:2181,broker2:2181,broker3:2181
```

### SSH 설정

#### ssh 설치

```
apt-get install ssh
```

```
vim /etc/ssh/sshd_config
// 38번째 줄의 PermitRootLogin의 주석을 해제
PermitsRootLogin yes
// 65번째 줄의 PasswordAuthentication를 yes로 변경한다
PasswordAuthentication yes
```

```
/etc/init.d/sshd restart

// 각각의 서버의 비밀번호를 설정해주자
passwd

//ssh-key를 생성한다 (모두 엔터를 누르자)
ssh-keygen
```

#### ssh 키 교환하기

모든 서버에서 아래의 명령어를 실행시켜주자

```
ssh-copy-id root@broker1
ssh-copy-id root@broker2
ssh-copy-id root@broker3
```


### 주키퍼 실행

주키퍼부터 실행해야한다. 주키퍼가 카프카의 메타데이터를 관리하기 때문이다
모든 브로커 서버에서 아래의 명령어를 실행시켜주자

```
# start
bin/zookeeper-server-start.sh -daemon config/zookeeper.properties

# stop
bin/zookeeper-server.stop.sh
```

### 카프카 실행

모든 브로커 서버에서 아래의 명령어를 실행시켜주자

```
# start
bin/kafka-server-start.sh -daemon config/server.properties

# stop
bin/kafka-server-stop.sh
```
### 클러스터 브로커 리스트 확인

```
bin/kafka-broker-api-versions.sh --bootstrap-server broker1:9092 | grep 9092
```

### HADOOP MASTER 서버와 KAFKA BROKER 연결

HADOOP MASTER 서버와 WORKER 서버에 카프카 브로커의 hostname을 등록해주자

```
vim /etc/hosts
(broker1 ip) broker1
(broker2 ip) broker2
(broker3 ip) broker3
```

이제 HADOOP MASTER 서버에서 KAFKA IP를 사용해 CONSUMER를 실행시킬 수 있다

**유의할 점**

1. 분산 시스템을 구축하기 위해서는 각각의 서버가 ssh를 통해 통신하여야한다
   kafka 분산시스템을 만들 때 ssh를 간과하여 한참 헤맸다
2. hostname을 가급적 먼저 수정하여 설정파일에 들어갈 hostname을 나중에 수정하지 않게끔 하자
3. 스칼라는 아직까진 2.12버전이 많이 쓰이는 것 같다(maven plugin도 2.12버전을 지원하는 것이 많다)
4. hadoop 분산 시스템을 초기화 할 때는 datanode 디렉토리를 완전히 삭제한 후 `hdfs namenode -format`을 실행시켜야 datanode와 namenode가 동일한 clusterID를 갖게된다
5. 각각의 클러스터의 hostname을 /etc/hosts에 ip주소와 함께 등록해야한다

**에러**

1. broker1의 이미지로 다른 브로커를 구성했을 때 다른 설정은 맞는데 커넥션이 이루어지지 않는 경우
   broker1에서 테스트 하지 말고 다른 브로커에서 zookeeper와 kafka 서버를 구동시켜보자(데몬X)
   `Configured broker.id 2 doesn't match stored broker.id Some(1) in meta.properties.`
   이 경우는 /tmp/kafka-logs/meta.properties에 broker.id가 변경되지 않아 생긴 문제였다
   따라서 rm /tmp/kafka-logs/meta.properties로 삭제시킨 후 재작동하였다
2. 하둡 클러스터와 카프카 브로커를 모두 구성한 뒤 spark-shell를 구동시킬 때 생긴 문제
   namenode host를 찾지 못하는데 spark-default.conf에서
   `spark.eventLog.enabled true spark.eventLog.dir hdfs://namenode:8021/spark_enginelog`
   를 주석처리하였다
3. ` spark.read.format("kafka").option("kafka.bootstrap.servers", "172.17.0.2:9092,172.17.0.3:9093,172.17.0.4:9094").option("subscribe", "bmt").option("startingOffsets","earliest").load()` 이후 .show로 브로커의 로그에 저장된 값을 읽으려 했지만 `UnknownHostException: broker2`가 발생
   master 서버에서 /etc/hosts에 broker ip 정보를 등록했지만 이번에는 진행창은 뜨지만 더이상 진행되지 않는 상황 발생
   worker 서버에도 똑같이 /etc/hosts에 broker ip 정보를 등록하니 정상 작동하였다
4. 외부에서 카프카에 데이터를 줄 때는 metadata라는 것을 kafka로부터 받게 되는데 이 때 metadata에는 config/server.properties파일의 advertise.listener에 적혀있는 주소를 넘겨준다
   이 때 주소가 ip가 아닌 hostname으로 되어있으면 외부에서는 찾아올 수 없으므로 꼭 ip주소로 적어놓자

**의문점**

1. server.properties의 advertise.listener는 콤마로 구분된 리스트라고 들었는데 두개를 넣었을때 변화가 없다... 첫번째 인자로 실패했을 때 두번째를 트라이하는 기능이 없는것인가?
