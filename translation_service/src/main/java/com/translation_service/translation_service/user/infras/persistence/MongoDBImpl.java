package com.translation_service.translation_service.user.infras.persistence;

import com.mongodb.client.MongoCollection;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.Updates;
import com.mongodb.client.result.UpdateResult;
import com.translation_service.translation_service.user.infras.Mapper.UserMapper;
import org.bson.Document;
import org.bson.conversions.Bson;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.data.mongodb.core.query.Update;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class MongoDBImpl {

    private final MongoTemplate mongoTemplate;
    private final UserMongoRepository userMongoRepository;

    public MongoDBImpl(MongoTemplate mongoTemplate, UserMongoRepository userMongoRepository) {
        this.mongoTemplate = mongoTemplate;
        this.userMongoRepository = userMongoRepository;
    }

    public Optional<UserDocument> findById(Integer userId) {
        System.out.println("Fetching user with ID: " + userId);

        // Query to find documents with the given userId in either 'before' or 'after'
        Query query = new Query(new Criteria().orOperator(
                Criteria.where("payload.before.id").is(userId),
                Criteria.where("payload.after.id").is(userId)
        ));
        System.out.println("Query: " + query);

        List<Document> docs = mongoTemplate.find(query, Document.class, "users");
        System.out.println("Documents found: " + docs);

        UserDocument userDoc = UserMapper.fromMongoDocument(docs);
        return Optional.ofNullable(userDoc);
    }



    public void addSentenceToUser(Long userId, String sentence){
        MongoCollection<Document> collection =
            mongoTemplate.getMongoDatabaseFactory().getMongoDatabase("my_mongo_db").getCollection("users");

        // Update query to find the user by ID
        Bson filter = Filters.eq("payload.after.id", userId);

        // Ensure the sentences field exists, then push the new sentence
        Bson update = Updates.push("payload.after.sentences", sentence);

        // Update operation
        // Perform the update
        UpdateResult result = collection.updateOne(filter, update);

        if (result.getMatchedCount() == 0) {
            throw new RuntimeException("User not found in MongoDB");
        }    }

//    public void createUser(String id, UserDocument user){
//        userMongoRepository.createUser(id, user);
//    }
}
