package com.translation_service.translation_service.user.domain.service;

import com.mongodb.client.MongoCollection;
import com.translation_service.translation_service.common.model.User;
import com.translation_service.translation_service.user.infras.Mapper.UserMapper;
import com.translation_service.translation_service.user.infras.persistence.MongoDBImpl;
import com.translation_service.translation_service.user.infras.persistence.UserDocument;
import com.translation_service.translation_service.user.infras.persistence.UserMongoRepository;
import org.bson.Document;
import org.springframework.stereotype.Service;

@Service
public class UserService{
    private final MongoDBImpl mongoDBImpl;
    private final UserMongoRepository userMongoRepository;
    private final UserMapper userMapper;

    public UserService(MongoDBImpl mongoDBImpl, UserMongoRepository userMongoRepository, UserMapper userMapper) {
        this.mongoDBImpl = mongoDBImpl;
        this.userMongoRepository = userMongoRepository;
        this.userMapper = userMapper;
    }

    public User getUser(Integer id) {
        System.out.println("Service hit: userId = " + id);
        return mongoDBImpl.findById(id)
                .map(userDoc -> UserMapper.toUser(userDoc)) // Correct method reference
                .orElseThrow(() -> new RuntimeException("User not found"));
    }


    public void addSentenceToUser(Long userId, String sentence) {
        System.out.println("Adding sentence: " + sentence + " to userID: " + userId);
        mongoDBImpl.addSentenceToUser(userId,sentence);
//        MongoCollection<Document> collection = mongoTemplate.getCollection("users");
//
//        // Update query to find the user by ID
//        Bson filter = Filters.eq("payload.after.id", userId);
//
//        // Ensure the sentences field exists, then push the new sentence
//        Bson update = Updates.push("payload.after.sentences", sentence);
//
//        // Update operation
//        collection.updateOne(filter, update);
    }

}