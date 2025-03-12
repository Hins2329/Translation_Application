# _Translation Application_
A microservices architecture, with core backend logic implemented using Spring
and Flask. Additionally, multiple Transformer models were integrated, and
PyTorch with CUDA was used to accelerate training.

## _Issues_:
* Due to mismanagement, the trained model and datasets have been lost.
    * JWTs cannot be passed between services due to a lack of familiarity with frontend HTML.
    * The pretrained model and related database have been lost due to losing access to my previous GitHub account.



## _Services & File Structure_
* Eureka_service (Eureka registration)
* Login And Register Service   
    Provides Login and Register function, store user info to MySQL, verify username/user-password
  *  Controller:   
     * UserController
  * Application:
     * LRapplication (Login and register application)
  * Config
     * MyBatisConfig
     * WebConfig
     * SecurityConfi
  * Domain.model
     * User 
  * infras
     * MybatisRepository 
  * Services
  * Util
    * Jwt
* Translation Service
    * Translate 
      * Controller 
      * Application
      * Domain.Feign
      * infra
    * User
* SyncDBs  
Configuration of Kafka+Debezium(Syncing DB from MySQL to MongoDB) and docker containers.
  * docker_compose.yml
  * DebeziumConfig
* Transformer_service  
  model.model hasn't benn uploaded due to some reasons.

[Login](login.pdf) 
[Redirecting](Redirecting.pdf)
[RegisterSuccess](RegisterSuccess.pdf)
[translatePage](translatePage.pdf)


